// https://stackoverflow.com/a/73731908
import { useCallback, useEffect, useState } from 'react';

type UseSingleAndDoubleClickOptions = {
  onSingleClick: () => void;
  onDoubleClick: () => void;
  latency?: number;
};

export function useSingleAndDoubleClick({
  onSingleClick,
  onDoubleClick,
  latency = 250,
}: UseSingleAndDoubleClickOptions): () => void {
  const [click, setClick] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (click === 1) {
        onSingleClick();
      }
      setClick(0);
    }, latency);

    if (click === 2) {
      onDoubleClick();
    }

    return () => clearTimeout(timer);
  }, [click, onDoubleClick, latency, onSingleClick]);

  const onClick = useCallback(() => setClick((prev) => prev + 1), []);

  return onClick;
}
