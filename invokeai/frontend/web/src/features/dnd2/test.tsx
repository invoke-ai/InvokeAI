/**
 * @jsxRuntime classic
 * @jsx jsx
 */
import Button from '@atlaskit/button/new';
import ImageIcon from '@atlaskit/icon/core/migration/image';
import { easeInOut } from '@atlaskit/motion/curves';
import { largeDurationMs, mediumDurationMs } from '@atlaskit/motion/durations';
import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForExternal, monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { containsFiles, getFiles } from '@atlaskit/pragmatic-drag-and-drop/external/file';
import { preventUnhandled } from '@atlaskit/pragmatic-drag-and-drop/prevent-unhandled';
import { token } from '@atlaskit/tokens';
// eslint-disable-next-line @atlaskit/ui-styling-standard/use-compiled -- Ignored via go/DSP-18766
import { css } from '@emotion/react';
import { bind } from 'bind-event-listener';
import { Fragment, memo, useCallback, useEffect, useRef, useState } from 'react';
import invariant from 'tiny-invariant';

import { GlobalStyles } from './util/global-styles';

const galleryStyles = css({
  display: 'flex',
  width: '70vw',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 'var(--grid)',
  flexWrap: 'wrap',
});
const imageStyles = css({
  display: 'block',
  // borrowing values from pinterest
  // ratio: 0.6378378378
  width: '216px',
  height: '340px',
  objectFit: 'cover',
});
const uploadStyles = css({
  // overflow: 'hidden',
  position: 'relative',
  // using these to hide the details
  borderRadius: 'calc(var(--grid) * 2)',
  overflow: 'hidden',
  transition: `opacity ${largeDurationMs}ms ${easeInOut}, filter ${largeDurationMs}ms ${easeInOut}`,
});
const loadingStyles = css({
  opacity: '0',
  filter: 'blur(1.5rem)',
});
const readyStyles = css({
  opacity: '1',
  filter: 'blur(0)',
});

const uploadDetailStyles = css({
  display: 'flex',
  boxSizing: 'border-box',
  width: '100%',
  padding: 'var(--grid)',
  position: 'absolute',
  bottom: 0,
  gap: 'var(--grid)',
  flexDirection: 'row',
  // background: token('color.background.sunken', fallbackColor),
  backgroundColor: 'rgba(255,255,255,0.5)',
});

const uploadFilenameStyles = css({
  flexGrow: '1',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
});

type UserUpload = {
  type: 'image';
  dataUrl: string;
  name: string;
  size: number;
};

const Upload = memo(function Upload({ upload }: { upload: UserUpload }) {
  const [state, setState] = useState<'loading' | 'ready'>('loading');
  const clearTimeout = useRef<() => void>(() => {});

  useEffect(function mount() {
    return function unmount() {
      clearTimeout.current();
    };
  }, []);

  return (
    <div css={[uploadStyles, state === 'loading' ? loadingStyles : readyStyles]}>
      <img
        src={upload.dataUrl}
        css={imageStyles}
        onLoad={() => {
          // this is the _only_ way I could find to get the animation to run
          // correctly every time in all browsers
          // setTimeout(fn, 0) -> sometimes wouldn't work in chrome (event nesting two)
          // requestAnimationFrame -> nope (event nesting two)
          // requestIdleCallback -> nope (doesn't work in safari)
          // I can find no reliable hook for applying the `ready` state,
          // this is the best I could manage 😩
          const timerId = setTimeout(() => setState('ready'), 100);
          clearTimeout.current = () => window.clearTimeout(timerId);
        }}
      />
      <div css={uploadDetailStyles}>
        <em css={uploadFilenameStyles}>{upload.name}</em>
        <code>{Math.round(upload.size / 1000)}kB</code>
      </div>
    </div>
  );
});

const Gallery = memo(function Gallery({ uploads: uploads }: { uploads: UserUpload[] }) {
  if (!uploads.length) {
    return null;
  }

  return (
    <div css={galleryStyles}>
      {uploads.map((upload, index) => (
        <Upload upload={upload} key={index} />
      ))}
    </div>
  );
});

const fileStyles = css({
  display: 'flex',
  flexDirection: 'column',
  padding: 'calc(var(--grid) * 6) calc(var(--grid) * 4)',
  boxSizing: 'border-box',
  alignItems: 'center',
  justifyContent: 'center',
  background: token('elevation.surface.sunken', '#091E4208'),
  borderRadius: 'var(--border-radius)',
  transition: `all ${mediumDurationMs}ms ${easeInOut}`,
  border: '2px dashed transparent',
  width: '100%',
  gap: token('space.300', '24px'),
});

const textStyles = css({
  color: token('color.text.disabled', '#091E424F'),
  fontSize: '1.4rem',
  display: 'flex',
  alignItems: 'center',
  gap: token('space.075'),
});

const overStyles = css({
  background: token('color.background.selected.hovered', '#CCE0FF'),
  color: token('color.text.selected', '#0C66E4'),
  borderColor: token('color.border.brand', '#0C66E4'),
});

const potentialStyles = css({
  borderColor: token('color.border.brand', '#0C66E4'),
});

const appStyles = css({
  display: 'flex',
  alignItems: 'center',
  gap: 'calc(var(--grid) * 2)',
  flexDirection: 'column',
});

const displayNoneStyles = css({ display: 'none' });

function Uploader() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<'idle' | 'potential' | 'over'>('idle');
  const [uploads, setUploads] = useState<UserUpload[]>([]);

  /**
   * Creating a stable reference so that we can use it in our unmount effect.
   *
   * If we used uploads as a dependency in the second `useEffect` it would run
   * every time the uploads changed, which is not desirable.
   */
  const stableUploadsRef = useRef<UserUpload[]>(uploads);
  useEffect(() => {
    stableUploadsRef.current = uploads;
  }, [uploads]);

  useEffect(() => {
    return () => {
      /**
       * MDN recommends explicitly releasing the object URLs when possible,
       * instead of relying just on the browser's garbage collection.
       */
      stableUploadsRef.current.forEach((upload) => {
        URL.revokeObjectURL(upload.dataUrl);
      });
    };
  }, []);

  const addUpload = useCallback((file: File | null) => {
    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      return;
    }

    const upload: UserUpload = {
      type: 'image',
      dataUrl: URL.createObjectURL(file),
      name: file.name,
      size: file.size,
    };
    setUploads((current) => [...current, upload]);
  }, []);

  const onFileInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.currentTarget.files ?? []);
      files.forEach(addUpload);
    },
    [addUpload]
  );

  useEffect(() => {
    const el = ref.current;
    invariant(el);
    return combine(
      dropTargetForExternal({
        element: el,
        canDrop: containsFiles,
        onDragEnter: () => setState('over'),
        onDragLeave: () => setState('potential'),
        onDrop: async ({ source }) => {
          const files = await getFiles({ source });

          files.forEach((file) => {
            if (file == null) {
              return;
            }
            if (!file.type.startsWith('image/')) {
              return;
            }
            const reader = new FileReader();
            reader.readAsDataURL(file);

            // for simplicity:
            // - not handling errors
            // - not aborting the
            // - not unbinding the event listener when the effect is removed
            bind(reader, {
              type: 'load',
              listener(event) {
                const result = reader.result;
                if (typeof result === 'string') {
                  const upload: UserUpload = {
                    type: 'image',
                    dataUrl: result,
                    name: file.name,
                    size: file.size,
                  };
                  setUploads((current) => [...current, upload]);
                }
              },
            });
          });
        },
      }),
      monitorForExternal({
        canMonitor: containsFiles,
        onDragStart: () => {
          setState('potential');
          preventUnhandled.start();
        },
        onDrop: () => {
          setState('idle');
          preventUnhandled.stop();
        },
      })
    );
  });

  /**
   * We trigger the file input manually when clicking the button. This also
   * works when selecting the button using a keyboard.
   *
   * We do this for two reasons:
   *
   * 1. Styling file inputs is very limited.
   * 2. Associating the button as a label for the input only gives us pointer
   *    support, but does not work for keyboard.
   */
  const inputRef = useRef<HTMLInputElement>(null);
  const onInputTriggerClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return (
    <div css={appStyles}>
      <div
        ref={ref}
        data-testid="drop-target"
        css={[fileStyles, state === 'over' ? overStyles : state === 'potential' ? potentialStyles : undefined]}
      >
        <strong css={textStyles}>
          Drop some images on me! <ImageIcon color="currentColor" spacing="spacious" label="" />
        </strong>

        <Button onClick={onInputTriggerClick}>Select images</Button>

        <input
          ref={inputRef}
          css={displayNoneStyles}
          id="file-input"
          onChange={onFileInputChange}
          type="file"
          accept="image/*"
          multiple
        />
      </div>
      <Gallery uploads={uploads} />
    </div>
  );
}

export default function Example() {
  return (
    <Fragment>
      <GlobalStyles />
      <Uploader />
    </Fragment>
  );
}
