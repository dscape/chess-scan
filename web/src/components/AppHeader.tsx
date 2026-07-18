interface AppHeaderProps {
  onReset?: () => void;
}

export default function AppHeader({ onReset }: AppHeaderProps) {
  return (
    <header className="app-header">
      <button type="button" className="brand" onClick={onReset} aria-label="Chess Scan home">
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
      </button>
      <div className="app-header__tag">Human-checked vision</div>
    </header>
  );
}
