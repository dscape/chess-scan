import {
  isMatchingPrimary,
  parseBestMove,
  parseInfoLine,
  type EngineLine,
} from "./uci";

export type AnalysisOptions = {
  budgetMs?: number;
  signal?: AbortSignal;
  onUpdate?: (line: EngineLine) => void;
};

export type AnalysisResult = {
  line: EngineLine | null;
};

type WorkerLike = Pick<Worker, "postMessage" | "terminate" | "onmessage" | "onerror">;
type ActiveAnalysis = {
  line: EngineLine | null;
  linesByMove: Map<string, EngineLine>;
  onUpdate?: (line: EngineLine) => void;
  resolve: (result: AnalysisResult) => void;
  reject: (reason: unknown) => void;
  drained: Promise<void>;
  drain: () => void;
  stopping: boolean;
  timeout: number;
};

type MessageWaiter = {
  expected: string;
  resolve: () => void;
  reject: (reason: unknown) => void;
  timeout: number;
};

const WORKER_URL = "/stockfish/stockfish-18-lite-single.js";
const STARTUP_TIMEOUT_MS = 60_000;

export default class StockfishClient {
  private worker: WorkerLike;
  private ready: Promise<void>;
  private waiters: MessageWaiter[] = [];
  private active: ActiveAnalysis | null = null;
  private disposed = false;

  constructor(createWorker: () => WorkerLike = () => new Worker(WORKER_URL)) {
    this.worker = createWorker();
    this.worker.onmessage = (event) => this.handleMessage(event.data);
    this.worker.onerror = (event) => this.fatal(
      new Error(event.message || "Stockfish failed to load."),
    );
    this.ready = this.initialize();
  }

  async analyze(fen: string, options: AnalysisOptions = {}): Promise<AnalysisResult> {
    const budgetMs = Math.min(Math.max(options.budgetMs ?? 1800, 250), 10_000);
    await this.ready;
    this.assertAvailable();
    await this.stop();
    if (options.signal?.aborted) throw abortError();

    this.send("ucinewgame");
    this.send(`position fen ${fen}`);

    let drain = () => {};
    const drained = new Promise<void>((resolve) => { drain = resolve; });
    const result = new Promise<AnalysisResult>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        if (this.active) this.fatal(new Error("Stockfish analysis timed out."));
      }, budgetMs + 5000);
      this.active = {
        line: null,
        linesByMove: new Map(),
        onUpdate: options.onUpdate,
        resolve,
        reject,
        drained,
        drain,
        stopping: false,
        timeout,
      };
    });

    const abort = () => { void this.stop(); };
    options.signal?.addEventListener("abort", abort, { once: true });
    this.send(`go movetime ${budgetMs}`);
    try {
      return await result;
    } finally {
      options.signal?.removeEventListener("abort", abort);
    }
  }

  async stop(): Promise<void> {
    const active = this.active;
    if (!active) return;
    if (!active.stopping) {
      active.stopping = true;
      this.send("stop");
    }
    await active.drained;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.fail(abortError());
    this.worker.terminate();
  }

  private async initialize(): Promise<void> {
    const uciReady = this.waitFor("uciok", STARTUP_TIMEOUT_MS);
    this.send("uci");
    await uciReady;
    this.send("setoption name Hash value 16");
    const engineReady = this.waitFor("readyok");
    this.send("isready");
    await engineReady;
  }

  private waitFor(expected: string, timeoutMs = 10_000): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        this.waiters = this.waiters.filter((waiter) => waiter.expected !== expected);
        reject(new Error(`Stockfish did not answer ${expected}.`));
      }, timeoutMs);
      this.waiters.push({ expected, resolve, reject, timeout });
    });
  }

  private handleMessage(data: unknown): void {
    for (const message of String(data).split(/\r?\n/)) {
      const line = message.trim();
      if (!line) continue;
      const waiter = this.waiters.find((candidate) => line === candidate.expected);
      if (waiter) {
        window.clearTimeout(waiter.timeout);
        this.waiters = this.waiters.filter((candidate) => candidate !== waiter);
        waiter.resolve();
        continue;
      }
      if (line.startsWith("id name ")) continue;

      const active = this.active;
      if (!active) continue;
      const info = parseInfoLine(line);
      if (info && !info.score.bound) {
        if (!active.line || info.depth >= active.line.depth) {
          active.line = info;
          active.onUpdate?.(info);
        }
        const firstMove = info.pv[0];
        if (firstMove) {
          const previous = active.linesByMove.get(firstMove);
          if (!previous || info.depth >= previous.depth) {
            active.linesByMove.set(firstMove, info);
          }
        }
        continue;
      }
      const bestMove = parseBestMove(line);
      if (bestMove) this.finishAnalysis(active, bestMove);
    }
  }

  private finishAnalysis(active: ActiveAnalysis, bestMove: string): void {
    if (this.active !== active) return;
    window.clearTimeout(active.timeout);
    this.active = null;
    active.drain();
    if (active.stopping) {
      active.reject(abortError());
      return;
    }
    if (bestMove === "(none)") {
      active.resolve({ line: null });
      return;
    }
    const line = isMatchingPrimary(bestMove, active.line)
      ? active.line
      : active.linesByMove.get(bestMove);
    if (!line) {
      active.reject(new Error("Stockfish did not return a matching principal variation."));
      return;
    }
    active.resolve({ line });
  }

  private send(command: string): void {
    this.assertAvailable();
    this.worker.postMessage(command);
  }

  private assertAvailable(): void {
    if (this.disposed) throw new Error("Stockfish has been disposed.");
  }

  private fatal(reason: unknown): void {
    if (!this.disposed) {
      this.disposed = true;
      this.worker.terminate();
    }
    this.fail(reason);
  }

  private fail(reason: unknown): void {
    for (const waiter of this.waiters) {
      window.clearTimeout(waiter.timeout);
      waiter.reject(reason);
    }
    this.waiters = [];
    const active = this.active;
    if (active) {
      window.clearTimeout(active.timeout);
      active.reject(reason);
      active.drain();
      this.active = null;
    }
  }
}

function abortError(): DOMException {
  return new DOMException("Stockfish analysis was cancelled.", "AbortError");
}
