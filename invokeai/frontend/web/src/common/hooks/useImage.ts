import { useLayoutEffect, useRef, useState } from 'react';

// Adapted from https://github.com/konvajs/use-image

type CrossOrigin = 'anonymous' | 'use-credentials';
type ReferrerPolicy =
  | 'no-referrer'
  | 'no-referrer-when-downgrade'
  | 'origin'
  | 'origin-when-cross-origin'
  | 'same-origin'
  | 'strict-origin'
  | 'strict-origin-when-cross-origin'
  | 'unsafe-url';
type ImageStatus = 'loaded' | 'loading' | 'failed';

export const useImage = (
  url: string,
  crossOrigin?: CrossOrigin,
  referrerpolicy?: ReferrerPolicy
): [undefined | HTMLImageElement, ImageStatus, Blob | null] => {
  // lets use refs for image and status
  // so we can update them during render
  // to have instant update in status/image when new data comes in
  const statusRef = useRef<ImageStatus>('loading');
  const imageRef = useRef<HTMLImageElement>();
  const blobRef = useRef<Blob | null>(null);

  // we are not going to use token
  // but we need to just to trigger state update
  const [_, setStateToken] = useState(0);

  // keep track of old props to trigger changes
  const oldUrl = useRef<string>();
  const oldCrossOrigin = useRef<string>();
  const oldReferrerPolicy = useRef<string>();

  if (
    oldUrl.current !== url ||
    oldCrossOrigin.current !== crossOrigin ||
    oldReferrerPolicy.current !== referrerpolicy
  ) {
    statusRef.current = 'loading';
    imageRef.current = undefined;
    oldUrl.current = url;
    oldCrossOrigin.current = crossOrigin;
    oldReferrerPolicy.current = referrerpolicy;
  }

  useLayoutEffect(
    function () {
      if (!url) {
        return;
      }
      const img = document.createElement('img');

      function onload() {
        statusRef.current = 'loaded';
        imageRef.current = img;
        const canvas = document.createElement('canvas');
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;

        const context = canvas.getContext('2d');
        if (context) {
          context.drawImage(img, 0, 0);
          canvas.toBlob(function (blob) {
            blobRef.current = blob;
          }, 'image/png');
        }
        setStateToken(Math.random());
      }

      function onerror() {
        statusRef.current = 'failed';
        imageRef.current = undefined;
        setStateToken(Math.random());
      }

      img.addEventListener('load', onload);
      img.addEventListener('error', onerror);
      if (crossOrigin) {
        img.crossOrigin = crossOrigin;
      }
      if (referrerpolicy) {
        img.referrerPolicy = referrerpolicy;
      }
      img.src = url;

      return function cleanup() {
        img.removeEventListener('load', onload);
        img.removeEventListener('error', onerror);
      };
    },
    [url, crossOrigin, referrerpolicy]
  );

  // return array because it is better to use in case of several useImage hooks
  // const [background, backgroundStatus] = useImage(url1);
  // const [patter] = useImage(url2);
  return [imageRef.current, statusRef.current, blobRef.current];
};
