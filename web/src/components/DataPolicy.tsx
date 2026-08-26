import { useEffect } from "react";

interface DataPolicyProps {
  diagnosticRetentionDays: number;
  onBack: () => void;
}

export default function DataPolicy({
  diagnosticRetentionDays,
  onBack,
}: DataPolicyProps) {
  useEffect(() => {
    const previousTitle = document.title;
    document.title = "Data use · Chess Scan";
    return () => {
      document.title = previousTitle;
    };
  }, []);

  return (
    <main className="data-policy-shell">
      <header className="data-policy-nav">
        <a
          className="brand"
          href="/"
          onClick={(event) => {
            event.preventDefault();
            onBack();
          }}
          aria-label="Back to Chess Scan"
        >
          <span className="brand__mark" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>Chess</strong>
            <em>Scan</em>
          </span>
        </a>
        <a
          className="data-policy-back"
          href="/"
          onClick={(event) => {
            event.preventDefault();
            onBack();
          }}
        >
          Back to scanner
        </a>
      </header>

      <article className="data-policy">
        <header className="data-policy__heading">
          <p className="eyebrow">Data use · Updated 26 August 2026</p>
          <h1>What happens to a scanned board.</h1>
          <p>
            Scanning works without sharing. Sharing stays off until you turn on
            <strong> Share clip</strong> in the camera, and that choice stays in this browser.
          </p>
        </header>

        <dl className="data-policy__summary" aria-label="Data-use summary">
          <div>
            <dt>Default</dt>
            <dd>Sharing off</dd>
          </div>
          <div>
            <dt>Camera clip</dt>
            <dd>10 seconds maximum</dd>
          </div>
          <div>
            <dt>Audio or screen</dt>
            <dd>Never recorded</dd>
          </div>
        </dl>

        <div className="data-policy__sections">
          <section>
            <span className="data-policy__number">01</span>
            <div>
              <h2>Every scan</h2>
              <p>
                Low-resolution camera previews and burst frames are sent to the scanner
                for board detection and classification. Supplemental frames are transient
                and are not stored individually. The full source photograph may show people,
                handwriting, or camera metadata, but it is temporary: it is deleted after you
                save, while abandoned scans expire automatically.
              </p>
              <p>
                A saved review keeps the final position, the raw model output and version,
                your corrections, the consent choice, and a random installation identifier.
                These records make accuracy and safety audits reproducible. When sharing is
                off, the rectified board image is deleted and the record is excluded from
                image training.
              </p>
            </div>
          </section>

          <section>
            <span className="data-policy__number">02</span>
            <div>
              <h2>When Share clip is on</h2>
              <p>
                After you correct and save a board, Chess Scan retains the
                perspective-corrected board crop, prediction, and final labels for model
                training and held-out evaluation. The original full photograph is still
                deleted.
              </p>
              <p>
                With the live camera, a supported browser also uploads up to ten seconds of
                camera-only video, capped at 8 MB, plus bounded browser, camera, timing,
                focus, and glare details. It may include anything visible around the board,
                so keep people and private material out of frame. The clip is deleted after
                {` ${diagnosticRetentionDays} days`} and is reviewed only to diagnose capture
                problems; it is not automatic model-training input.
              </p>
              <p>
                Choosing an existing photo cannot create a camera clip. If sharing was
                already on, only its corrected board crop and labels are eligible for
                training.
              </p>
            </div>
          </section>

          <section>
            <span className="data-policy__number">03</span>
            <div>
              <h2>Your control</h2>
              <p>
                The reusable on/off preference is stored only in this browser. Turn it off
                from the same camera control before a future scan. Turning it off does not rewrite an
                earlier confirmed feedback event or remove data already admitted to a
                training set.
              </p>
              <p>
                Chess Scan does not ask for a name, account, microphone, location, or screen
                access. The anonymous installation identifier is used to group duplicate
                contributions and prevent one device from dominating evaluation. Aggregate
                operational reports do not export images, labels, recordings, or installation
                identifiers.
              </p>
            </div>
          </section>
        </div>

        <footer className="data-policy__footer">
          <p>This page describes scanner data handling, not the contents of sites opened from a review.</p>
          <a
            href="https://github.com/dscape/chess-scan/blob/main/docs/camera-scanner.md"
            target="_blank"
            rel="noreferrer"
          >
            Read the technical data contract
          </a>
        </footer>
      </article>
    </main>
  );
}
