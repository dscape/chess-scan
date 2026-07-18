import { useEffect, useRef } from "react";
import type { ScanResult } from "../types";

interface RecognitionSuccessProps {
  scan: ScanResult;
  onContinue: () => void;
}

export default function RecognitionSuccess({ scan, onContinue }: RecognitionSuccessProps) {
  const onContinueRef = useRef(onContinue);
  onContinueRef.current = onContinue;

  useEffect(() => {
    const timer = window.setTimeout(() => onContinueRef.current(), 1500);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <main className="recognition-shell" aria-live="polite">
      <section className="recognition-card">
        <div className="recognition-board">
          <img src={scan.rectified_image_url} alt="Identified chess board" />
          <div className="recognition-board__green" aria-hidden="true" />
          <div className="recognition-board__grid" aria-hidden="true" />
          <span className="recognition-board__check" aria-hidden="true">✓</span>
        </div>
        <div className="recognition-copy">
          <p className="eyebrow">Board identified</p>
          <h1>64 squares<br />in place.</h1>
          <p>The diagram is straightened and ready for a human check.</p>
          <div className="recognition-progress" aria-hidden="true"><i /></div>
          <button type="button" className="primary-button" onClick={onContinue}>
            <span>Check the position</span>
            <span aria-hidden="true">→</span>
          </button>
        </div>
      </section>
    </main>
  );
}
