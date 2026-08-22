import { useState } from "react";
import axios from "axios";

export default function DocumentUpload({ onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://localhost:8000/documents/upload",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      onUploadComplete(res.data);
      setFile(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".pdf,.docx"
        onChange={(e) => setFile(e.target.files[0])}
        disabled={uploading}
      />
      <div style={{ marginTop: "0.75rem" }}>
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? "Ingesting..." : "Upload"}
        </button>
      </div>
      {error && <p style={{ color: "var(--danger)", fontSize: "0.85rem" }}>{error}</p>}
    </div>
  );
}