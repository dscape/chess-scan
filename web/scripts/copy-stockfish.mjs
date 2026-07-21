import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = join(root, "node_modules", "stockfish");
const output = join(root, "public", "stockfish");
const artifacts = [
  {
    source: "bin/stockfish-18-lite-single.js",
    sha256: "5243fd9b276cab7dfe3ad1d43ab9ead73568fac76468c614242977a210c4a391",
  },
  {
    source: "bin/stockfish-18-lite-single.wasm",
    sha256: "a8fbc05ec6920b56d7485826dcb02c5ffd2826bcbf751cf973046f237a9096f1",
  },
  { source: "Copying.txt" },
];

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
const hashes = {};
for (const artifact of artifacts) {
  const name = basename(artifact.source);
  if (artifact.sha256) {
    const bytes = await readFile(join(packageRoot, artifact.source));
    const actual = createHash("sha256").update(bytes).digest("hex");
    if (actual !== artifact.sha256) {
      throw new Error(`Unexpected SHA-256 for ${name}: ${actual}`);
    }
    hashes[name] = actual;
  }
  await copyFile(join(packageRoot, artifact.source), join(output, name));
}

const source = `# Stockfish browser engine source\n\nChess Scan serves the Stockfish worker as a separate GPLv3 component. Chess Scan's own source remains MIT-licensed.\n\n- npm package: stockfish@18.0.8\n- npm git commit: 93c994592dcf3b4b21052ab925e9b534df9c0918\n- corresponding source: https://github.com/nmrugg/stockfish.js/tree/93c994592dcf3b4b21052ab925e9b534df9c0918\n- corresponding source archive: https://github.com/nmrugg/stockfish.js/archive/93c994592dcf3b4b21052ab925e9b534df9c0918.tar.gz\n- package archive: https://registry.npmjs.org/stockfish/-/stockfish-18.0.8.tgz\n- package integrity: sha512-z+f2UMPXLylDBGjv9e9zU8QulY7hUl8MYHesLRrdddewlOXjJrUSmtNmbtID1/F72EPhq0CCkCNxgWS5MQVWtQ==\n\n## Served artifacts\n\n${Object.entries(hashes).map(([name, hash]) => `- \`${name}\`: SHA-256 \`${hash}\``).join("\n")}\n\nThe complete GPLv3 license is served beside these files as \`Copying.txt\`.\n`;
await writeFile(join(output, "SOURCE.md"), source);
