import {
  contiguousRankedLines,
  isStableLine,
  latestExactLine,
  parseBestMove,
  parseInfoLine,
  type EngineLine,
} from "./uci";

export type AnalysisOptions = {
  budgetMs?: number;
  multiPv?: number;
  newGame?: boolean;
  requireStable?: boolean;
  searchMoves?: string[];
  signal?: AbortSignal;
  onUpdate?: (lines: EngineLine[]) => void;
};

export type AnalysisResult = {
  lines: EngineLine[];
};

export class StockfishError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StockfishError";
  }
}

type WorkerLike = Pick<Worker, "postMessage" | "terminate" | "onmessage" | "onerror">;
type ActiveAnalysis = {
  lines: Map<number, EngineLine>;
  firstMovesByRank: Map<number, Map<number, string>>;
  requireStable: boolean;
  onUpdate?: (lines: EngineLine[]) => void;
  resolve: (result: AnalysisResult) => void;
  reject: (reason: unknown) => void;
  drained: Promise<void>;
  drain: () => void;
  minimumBudgetReached: boolean;
  stopping: boolean;
  stopRequested: boolean;
  stabilityTimeout: number | null;
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
      new StockfishError(event.message || "Stockfish failed to load."),
    );
    this.ready = this.initialize();
  }

  async analyze(fen: string, options: AnalysisOptions = {}): Promise<AnalysisResult> {
    const budgetMs = Math.min(Math.max(options.budgetMs ?? 1800, 250), 10_000);
    const multiPv = Math.min(Math.max(options.multiPv ?? 3, 1), 3);
    const requireStable = options.requireStable ?? true;
    const searchBudgetMs = requireStable ? Math.min(budgetMs * 2, 10_000) : budgetMs;
    await this.ready;
    this.assertAvailable();
    await this.stop();
    if (options.signal?.aborted) throw abortError();

    if (options.newGame) this.send("ucinewgame");
    this.send(`setoption name MultiPV value ${multiPv}`);
    this.send("setoption name UCI_ShowWDL value true");
    this.send(`position fen ${fen}`);

    let drain = () => {};
    const drained = new Promise<void>((resolve) => { drain = resolve; });
    const result = new Promise<AnalysisResult>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        if (this.active) this.fatal(new StockfishError("Stockfish analysis timed out."));
      }, searchBudgetMs + 5000);
      this.active = {
        lines: new Map(),
        firstMovesByRank: new Map(),
        requireStable,
        onUpdate: options.onUpdate,
        resolve,
        reject,
        drained,
        drain,
        minimumBudgetReached: false,
        stopping: false,
        stopRequested: false,
        stabilityTimeout: null,
        timeout,
      };
    });

    const active = this.active;
    if (requireStable && searchBudgetMs > budgetMs && active) {
      active.stabilityTimeout = window.setTimeout(() => {
        if (this.active !== active) return;
        active.minimumBudgetReached = true;
        this.stopIfStable(active);
      }, budgetMs);
    }

    const abort = () => { void this.stop(); };
    options.signal?.addEventListener("abort", abort, { once: true });
    const searchMoves = options.searchMoves?.length
      ? ` searchmoves ${options.searchMoves.join(" ")}`
      : "";
    this.send(`go movetime ${searchBudgetMs}${searchMoves}`);
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
      if (!active.stopRequested) {
        active.stopRequested = true;
        this.send("stop");
      }
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
        reject(new StockfishError(`Stockfish did not answer ${expected}.`));
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
        const latest = latestExactLine(active.lines.get(info.rank), info);
        if (latest !== info) continue;
        active.lines.set(info.rank, latest);
        if (info.pv[0]) {
          const history = active.firstMovesByRank.get(info.rank) ?? new Map<number, string>();
          history.set(info.depth, info.pv[0]);
          active.firstMovesByRank.set(info.rank, history);
        }
        active.onUpdate?.(orderedLines(active.lines));
        this.stopIfStable(active);
        continue;
      }
      const bestMove = parseBestMove(line);
      if (bestMove) this.finishAnalysis(active, bestMove);
    }
  }

  private stopIfStable(active: ActiveAnalysis): void {
    if (
      this.active !== active
      || !active.minimumBudgetReached
      || active.stopping
      || active.stopRequested
    ) return;
    const primary = active.lines.get(1);
    const expectedMove = primary?.pv[0];
    if (!expectedMove) return;
    const recentMoves = [...(active.firstMovesByRank.get(1)?.entries() ?? [])]
      .sort(([left], [right]) => right - left)
      .slice(0, 3)
      .map(([, move]) => move);
    if (!isStableLine(expectedMove, primary, recentMoves)) return;
    active.stopRequested = true;
    this.send("stop");
  }

  private finishAnalysis(active: ActiveAnalysis, bestMove: string): void {
    if (this.active !== active) return;
    window.clearTimeout(active.timeout);
    if (active.stabilityTimeout !== null) window.clearTimeout(active.stabilityTimeout);
    this.active = null;
    const allLines = orderedLines(active.lines);
    const primaryDepth = allLines.find((line) => line.rank === 1)?.depth ?? 0;
    const eligible = allLines.filter(
      (line) => line.depth >= primaryDepth - 2 && !line.score.bound && line.wdl,
    );
    const lines = contiguousRankedLines(eligible);
    active.drain();
    if (active.stopping) {
      active.reject(abortError());
      return;
    }
    if (bestMove === "(none)") {
      active.resolve({ lines: [] });
      return;
    }
    const primary = lines.find((line) => line.rank === 1);
    if (!primary || primary.pv[0] !== bestMove) {
      active.reject(new StockfishError("Stockfish did not return a matching principal variation."));
      return;
    }
    const markedLines = lines.map((line) => {
      const recentMoves = [...(active.firstMovesByRank.get(line.rank)?.entries() ?? [])]
        .sort(([left], [right]) => right - left)
        .slice(0, 3)
        .map(([, move]) => move);
      const expectedMove = line.rank === 1 ? bestMove : line.pv[0];
      return {
        ...line,
        stable: expectedMove !== undefined && isStableLine(expectedMove, line, recentMoves),
      };
    });
    if (active.requireStable && !markedLines[0]?.stable) {
      active.reject(new StockfishError("Stockfish did not return a stable principal variation."));
      return;
    }
    active.resolve({
      lines: active.requireStable
        ? contiguousRankedLines(markedLines.filter((line) => line.stable))
        : markedLines,
    });
  }

  private send(command: string): void {
    this.assertAvailable();
    this.worker.postMessage(command);
  }

  private assertAvailable(): void {
    if (this.disposed) throw new StockfishError("Stockfish has been disposed.");
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
      if (active.stabilityTimeout !== null) window.clearTimeout(active.stabilityTimeout);
      active.reject(reason);
      active.drain();
      this.active = null;
    }
  }
}

function orderedLines(lines: Map<number, EngineLine>): EngineLine[] {
  return [...lines.values()].sort((left, right) => left.rank - right.rank);
}

function abortError(): DOMException {
  return new DOMException("Stockfish analysis was cancelled.", "AbortError");
}
