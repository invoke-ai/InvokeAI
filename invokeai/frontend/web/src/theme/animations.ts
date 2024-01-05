import { keyframes } from '@emotion/react';

export const spinKeyframes = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

export const spinAnimation = `${spinKeyframes} 0.45s linear infinite`;

export const spinAnimationSlow = `${spinKeyframes} 1s linear infinite`;
