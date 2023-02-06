import { RefObject, useEffect } from 'react';

const watchers: {
  ref: RefObject<HTMLElement>;
  enable: boolean;
  callback: () => void;
}[] = [];

const useClickOutsideWatcher = () => {
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      watchers.forEach(({ ref, enable, callback }) => {
        if (enable && ref.current && !ref.current.contains(e.target as Node)) {
          callback();
        }
      });
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return {
    addWatcher: (watcher: {
      ref: RefObject<HTMLElement>;
      callback: () => void;
      enable: boolean;
    }) => {
      watchers.push(watcher);
    },
  };
};

export default useClickOutsideWatcher;
