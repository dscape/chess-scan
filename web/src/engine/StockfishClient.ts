import {
  isStablePrimary,
  parseBestMove,
  parseInfoLine,
  type EngineLine,
} from "./uci";

export type AnalysisOptions = {
  budgetMs?: number;
  multiPv?: number;
  signal?: AbortSignal;
  onUpdate?: (lines: EngineLine[]) => void;
};

export type AnalysisResult = {
  lines: EngineLine[];
};

type WorkerLike = Pick<Worker, "postMessage" | "terminate" | "onmessage" | "onerror">;
type ActiveAnalysis = {
  lines: Map<number, EngineLine>;
  primaryMoves: Map<number, string>;
  onUpdate?: (lines: EngineLine[]) => void;
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
    const multiPv = Math.min(Math.max(options.multiPv ?? 3, 1), 5);
    await this.ready;
    this.assertAvailable();
    await this.stop();
    if (options.signal?.aborted) throw abortError();

    this.send("ucinewgame");
    this.send(`setoption name MultiPV value ${multiPv}`);
    this.send("setoption name UCI_ShowWDL value true");
    this.send(`position fen ${fen}`);

    let drain = () => {};
    const drained = new Promise<void>((resolve) => { drain = resolve; });
    const result = new Promise<AnalysisResult>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        if (this.active) this.fatal(new Error("Stockfish analysis timed out."));
      }, budgetMs + 5000);
      this.active = {
        lines: new Map(),
        primaryMoves: new Map(),
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
      if (info) {
        const previous = active.lines.get(info.multipv);
        if (!previous || info.depth >= previous.depth) active.lines.set(info.multipv, info);
        if (info.multipv === 1 && !info.score.bound && info.pv[0]) {
          active.primaryMoves.set(info.depth, info.pv[0]);
        }
        active.onUpdate?.(orderedLines(active.lines));
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
    const allLines = orderedLines(active.lines);
    const primaryDepth = allLines.find((line) => line.multipv === 1)?.depth ?? 0;
    const lines = allLines.filter(
      (line) => line.depth >= primaryDepth - 2 && !line.score.bound,
    );
    active.drain();
    if (active.stopping) {
      active.reject(abortError());
      return;
    }
    if (bestMove === "(none)") {
      active.resolve({ lines: [] });
      return;
    }
    const primary = lines.find((line) => line.multipv === 1);
    const recentPrimaryMoves = [...active.primaryMoves.entries()]
      .sort(([left], [right]) => right - left)
      .slice(0, 3)
      .map(([, move]) => move);
    if (!isStablePrimary(bestMove, primary, recentPrimaryMoves)) {
      active.reject(new Error("Stockfish did not return a stable principal variation."));
      return;
    }
    active.resolve({ lines });
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

function orderedLines(lines: Map<number, EngineLine>): EngineLine[] {
  return [...lines.values()].sort((left, right) => left.multipv - right.multipv);
}

function abortError(): DOMException {
  return new DOMException("Stockfish analysis was cancelled.", "AbortError");
}
