import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { OpenAPI } from 'services/api';

export const getUrlAlt = (url: string, shouldTransformUrls: boolean) => {
  if (OpenAPI.BASE && shouldTransformUrls) {
    return [OpenAPI.BASE, url].join('/');
  }

  return url;
};

export const useGetUrl = () => {
  const shouldTransformUrls = useAppSelector(
    (state: RootState) => state.config.shouldTransformUrls
  );

  const getUrl = useCallback(
    (url?: string) => {
      if (OpenAPI.BASE && shouldTransformUrls) {
        return [OpenAPI.BASE, url].join('/');
      }

      return url;
    },
    [shouldTransformUrls]
  );

  return {
    shouldTransformUrls,
    getUrl,
  };
};
