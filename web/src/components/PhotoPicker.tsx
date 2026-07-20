import { useEffect, useRef, type ReactNode } from "react";

interface PhotoPickerProps {
  children: ReactNode;
  className: string;
  disabled?: boolean;
  onOpen?: () => void;
  onCancel?: () => void;
  onPhoto: (file: File) => void;
}

export default function PhotoPicker({
  children,
  className,
  disabled = false,
  onOpen,
  onCancel,
  onPhoto,
}: PhotoPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const input = inputRef.current;
    if (!input || !onCancel) return;
    const handleCancel = () => onCancel();
    input.addEventListener("cancel", handleCancel);
    return () => input.removeEventListener("cancel", handleCancel);
  }, [onCancel]);

  return (
    <label className={`${className} photo-picker${disabled ? " is-disabled" : ""}`}>
      {children}
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        disabled={disabled}
        onClick={onOpen}
        onChange={(event) => {
          const input = event.currentTarget;
          const file = input.files?.[0];
          input.value = "";
          if (file) onPhoto(file);
        }}
      />
    </label>
  );
}
