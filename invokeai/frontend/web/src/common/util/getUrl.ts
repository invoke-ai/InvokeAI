import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
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

  return {
    shouldTransformUrls,
    getUrl: (url?: string) => {
      if (OpenAPI.BASE && shouldTransformUrls) {
        return [OpenAPI.BASE, url].join('/');
      }

      return url;
    },
  };
};
