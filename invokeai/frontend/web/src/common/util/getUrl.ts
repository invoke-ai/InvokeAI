import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { OpenAPI } from 'services/api';

export const useGetUrl = () => {
  const shouldTransformUrls = useAppSelector(
    (state: RootState) => state.system.shouldTransformUrls
  );

  return {
    shouldTransformUrls,
    getUrl: (url: string) => {
      if (OpenAPI.BASE && shouldTransformUrls) {
        console.log('transformed');
        return [OpenAPI.BASE, url].join('/');
      }

      console.log('didnt transform');
      return url;
    },
  };
};
