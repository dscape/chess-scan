import type { ReviewBadge } from "../types";

type ReviewGlyphProps = {
  badge: ReviewBadge;
  className?: string;
};

export default function ReviewGlyph({ badge, className }: ReviewGlyphProps) {
  return (
    <svg
      className={["review-glyph", className].filter(Boolean).join(" ")}
      viewBox="-1 -1 26 26"
      aria-hidden="true"
    >
      <ReviewGlyphLayers badge={badge} />
    </svg>
  );
}

export function ReviewGlyphLayers({ badge, className }: ReviewGlyphProps) {
  return (
    <g
      className={["review-glyph-layers", className].filter(Boolean).join(" ")}
      fill="none"
    >
      <g className="review-glyph-layers__keyline">
        <ReviewGlyphPaths badge={badge} />
      </g>
      <g className="review-glyph-layers__face">
        <ReviewGlyphPaths badge={badge} />
      </g>
    </g>
  );
}

function ReviewGlyphPaths({ badge }: { badge: ReviewBadge }) {
  switch (badge) {
    case "fork":
      return (
        <>
          <path d="M12 20v-8M12 12 6.5 6.5M12 12l5.5-5.5" />
          <circle cx="6" cy="6" r="1.5" />
          <circle cx="18" cy="6" r="1.5" />
          <circle cx="12" cy="20" r="1.5" />
        </>
      );
    case "pin":
      return (
        <>
          <path d="m8 4 8 .1-1.7 5.2 2.9 2.9-10.4-.1 2.9-2.8L8 4Z" />
          <path d="M12 12.2V21" />
        </>
      );
    case "xray":
      return (
        <>
          <path d="M3.5 12s3.2-5 8.5-5 8.5 5 8.5 5-3.2 5-8.5 5-8.5-5-8.5-5Z" />
          <circle cx="12" cy="12" r="2.4" />
        </>
      );
    case "trap":
      return (
        <>
          <path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" />
          <path d="m9 9 6 6M15 9l-6 6" />
        </>
      );
    case "capture":
      return (
        <>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 3v5M12 16v5M3 12h5M16 12h5" />
        </>
      );
    case "clearance":
      return (
        <>
          <path d="M10 12H3M6 9l-3 3 3 3M14 12h7M18 9l3 3-3 3" />
          <path d="M12 5v14" strokeDasharray="2 2" />
        </>
      );
    case "discovery":
      return (
        <>
          <circle cx="12" cy="12" r="3" />
          <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" />
        </>
      );
    case "interference":
      return (
        <>
          <path d="M4 18 9.5 12.5M14.5 7.5 20 2" />
          <path d="m7 5 12 12M9.5 9.5l5 5" />
        </>
      );
    case "attraction":
      return (
        <>
          <path d="M7 4v8a5 5 0 0 0 10 0V4" />
          <path d="M7 4h4v4H7M13 4h4v4h-4" />
        </>
      );
    case "intermezzo":
      return <path d="m13.5 2-7 11H12l-1.5 9 7-12H12l1.5-8Z" />;
    case "mate":
      return (
        <>
          <path d="M8 4 6 20M16 4l-2 16M4 9h16M3 15h16" />
        </>
      );
    case "engine":
      return (
        <>
          <path d="M12 3 13.8 9.2 20 11l-6.2 1.8L12 19l-1.8-6.2L4 11l6.2-1.8L12 3Z" />
        </>
      );
  }
}
