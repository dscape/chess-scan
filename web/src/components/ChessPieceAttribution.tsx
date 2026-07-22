type ChessPieceAttributionProps = {
  className?: string;
};

export default function ChessPieceAttribution({
  className,
}: ChessPieceAttributionProps) {
  return (
    <span className={className}>
      Adapted from{" "}
      <a href={SOURCE_URL} target="_blank" rel="noreferrer">
        Chess Simple Assets by Maciej Świerczek
      </a>{" "}
      ·{" "}
      <a href={LICENSE_URL} target="_blank" rel="noreferrer">
        CC BY 4.0
      </a>
    </span>
  );
}

const SOURCE_URL =
  "https://www.figma.com/community/file/971870797656870866/chess-simple-assets";
const LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/";
