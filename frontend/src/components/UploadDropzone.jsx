import { useRef, useState } from "react";

export default function UploadDropzone({
  multiple = false,
  title,
  description,
  onFilesSelected,
  disabled = false
}) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (fileList) => {
    const files = Array.from(fileList || []);
    if (files.length) {
      onFilesSelected(files);
    }
  };

  return (
    <div
      className={`dropzone ${isDragging ? "dropzone--active" : ""} ${
        disabled ? "dropzone--disabled" : ""
      }`}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) {
          setIsDragging(true);
        }
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        event.preventDefault();
        setIsDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        if (!disabled) {
          handleFiles(event.dataTransfer.files);
        }
      }}
    >
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        multiple={multiple}
        disabled={disabled}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />

      <div className="dropzone__icon" aria-hidden="true">
        <span />
      </div>
      <h2 className="dropzone__title">{title}</h2>
      <p className="dropzone__text">{description}</p>
      <div className="dropzone__actions">
        <button
          type="button"
          className="button button--primary"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          Choose files
        </button>
        <span className="dropzone__hint">
          {multiple ? "Supports multiple images" : "Upload a single face image"}
        </span>
      </div>
    </div>
  );
}
