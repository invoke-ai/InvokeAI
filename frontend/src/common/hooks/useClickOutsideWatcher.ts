import { RefObject, useEffect } from 'react';

const useClickOutsideWatcher = (
  ref: RefObject<HTMLElement>,
  callback: () => void,
  req = true
) => {
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        callback();
      }
    }
    if (req) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      if (req) {
        document.removeEventListener('mousedown', handleClickOutside);
      }
    };
  }, [ref, req, callback]);
};

export default useClickOutsideWatcher;
