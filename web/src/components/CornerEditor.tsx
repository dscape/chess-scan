import { useRef, useState, type PointerEvent } from "react";
import type { Point } from "../types";

interface CornerEditorProps {
  imageUrl: string;
  width: number;
  height: number;
  corners: Point[];
  onChange: (corners: Point[]) => void;
  disabled?: boolean;
}

export default function CornerEditor({
  imageUrl,
  width,
  height,
  corners,
  onChange,
  disabled = false,
}: CornerEditorProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const handleRadius = Math.max(width, height) * 0.018;
  const hitRadius = Math.max(width, height) * 0.055;

  function startDrag(index: number, event: PointerEvent<SVGCircleElement>) {
    if (disabled) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(index);
  }

  function moveDrag(event: PointerEvent<SVGSVGElement>) {
    if (dragging === null || disabled) return;
    const point = pointFromEvent(event);
    if (!point) return;
    const next = corners.map((corner, index) => (index === dragging ? point : corner));
    onChange(next);
  }

  function pointFromEvent(event: PointerEvent<SVGSVGElement>): Point | null {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const transformed = point.matrixTransform(matrix.inverse());
    return [
      Math.max(0, Math.min(width - 1, transformed.x)),
      Math.max(0, Math.min(height - 1, transformed.y)),
    ];
  }

  return (
    <svg
      ref={svgRef}
      className={`corner-editor ${disabled ? "is-disabled" : ""}`}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Photograph with four adjustable board corners"
      onPointerMove={moveDrag}
      onPointerUp={() => setDragging(null)}
      onPointerCancel={() => setDragging(null)}
    >
      <image href={imageUrl} width={width} height={height} preserveAspectRatio="none" />
      <polygon
        points={corners.map(([x, y]) => `${x},${y}`).join(" ")}
        className="corner-editor__board"
      />
      {corners.map(([x, y], index) => (
        <g key={index}>
          <circle
            cx={x}
            cy={y}
            r={hitRadius}
            className="corner-editor__hit-target"
            onPointerDown={(event) => startDrag(index, event)}
          />
          <circle
            cx={x}
            cy={y}
            r={handleRadius * 1.8}
            className="corner-editor__halo"
          />
          <circle
            cx={x}
            cy={y}
            r={handleRadius}
            className="corner-editor__handle"
            onPointerDown={(event) => startDrag(index, event)}
          />
          <text
            x={x}
            y={y + handleRadius * 0.34}
            className="corner-editor__number"
            textAnchor="middle"
          >
            {index + 1}
          </text>
        </g>
      ))}
    </svg>
  );
}
