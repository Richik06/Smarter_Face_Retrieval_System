import { useEffect, useState } from "react";

export function useSessionState(key, initialValue) {
  const [state, setState] = useState(() => {
    try {
      const stored = window.sessionStorage.getItem(key);
      return stored ? JSON.parse(stored) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.sessionStorage.setItem(key, JSON.stringify(state));
    } catch {
      // Ignore storage failures in private or restricted contexts.
    }
  }, [key, state]);

  return [state, setState];
}
