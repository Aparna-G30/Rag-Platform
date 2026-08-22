export default function SourceCard({ source, index }) {
  return (
    <div className="callout">
      <div className="callout-num">{index + 1}</div>
      <span className="callout-page">PG. {source.page_number}</span>
      {source.content.slice(0, 200)}...
    </div>
  );
}