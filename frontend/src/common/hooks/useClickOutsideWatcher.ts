import { RefObject, useEffect } from 'react';

const useClickOutsideWatcher = (
  ref: RefObject<HTMLElement>,
  callback: () => void
) => {
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        callback();
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [ref, callback]);
};

export default useClickOutsideWatcher;
