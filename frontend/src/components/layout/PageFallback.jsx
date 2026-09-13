export function PageFallback() {
  return (
    <div className="boot">
      <div className="boot__card">
        <span className="spinner spinner--lg" />
        <div>
          <div className="strong">Loading…</div>
          <div className="tiny muted">Fetching this workspace section</div>
        </div>
      </div>
    </div>
  );
}

export default PageFallback;
