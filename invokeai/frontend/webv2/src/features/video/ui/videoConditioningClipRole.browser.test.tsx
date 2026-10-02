import type { GalleryUiAdapter } from '@features/gallery/react';
import type { VideoConditioningClip } from '@features/video/core/types';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryUiProvider } from '@features/gallery/react';
import { VideoUiProvider } from '@features/video/ui/VideoUiContext';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoConditioningClipField } from './VideoConditioningClipField';

/**
 * The role decides which half of the clip the model is given and which it generates -- the whole
 * difference between audio-to-video and video-to-audio. It reaches the panel through one Select
 * whose value is an array, so a wrong-shaped value shows the placeholder while the stored role
 * quietly stays whatever it was: a control that looks unset and generates the other mode.
 */

const CLIP: VideoConditioningClip = {
  clip: { fps: 24, height: 704, numFrames: 96, video_name: 'clip.mp4', width: 1248 },
  fpsKnown: true,
  role: 'audio',
};

const noop = vi.fn();
/** The media slot reaches for the gallery adapter on render; nothing here exercises it. */
const galleryAdapter = {
  ItemActionsProvider: ({ children }: { children: ReactNode }) => children,
  ImageContextMenu: () => null,
  antialiasProgressImages: false,
  exportProject: noop,
  followProgressSession: noop,
  followedProgressSessionId: null,
  gallery: new Proxy({}, { get: () => noop }),
  galleryValues: {},
  generateValues: {},
  getItemLabel: () => Promise.resolve(null),
  liveFollowEnabled: false,
  notifications: { add: noop, reportError: noop },
  pinnedProgressSessionId: null,
  progressSessions: [],
  projectId: 'project-1',
  projectName: 'Project',
  widgets: { openGallery: () => true, patchGalleryValues: noop },
} as unknown as GalleryUiAdapter;

/** The field reaches for find-in-gallery on render; nothing here exercises it. */
const videoAdapter = {
  findInGallery: noop,
  getUploadBoardId: () => 'none',
  patchValues: noop,
  playVideoSpanInPreview: () => null,
  projectId: 'project-1',
  rawValues: {},
  reportError: noop,
  showPromptSyntaxHighlighting: false,
  touchGalleryImages: noop,
  videoSpanPlayback: { getState: () => null, subscribe: () => () => undefined },
};

const i18n = createInstance();

void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          video: {
            conditioningClipHelp: 'Drop a clip to generate the other half of it.',
            conditioningRole: 'Use from this clip',
            conditioningRoleAudio: 'Its soundtrack',
            conditioningRoleVideo: 'Its picture',
            conditioningRoleVideoHelp: 'The clip is re-encoded at the canvas.',
          },
        },
      },
    },
  },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('the conditioning clip role control', () => {
  let host: HTMLDivElement;
  let root: Root;
  const onChange = vi.fn();
  const mount = (conditioningClip: VideoConditioningClip | null, disabled?: { reason: string }) =>
    act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <GalleryUiProvider adapter={galleryAdapter}>
              <VideoUiProvider adapter={videoAdapter}>
                <DndContext>
                  <VideoConditioningClipField
                    conditioningClip={conditioningClip}
                    disabled={Boolean(disabled)}
                    disabledReason={disabled?.reason}
                    onChange={onChange}
                  />
                </DndContext>
              </VideoUiProvider>
            </GalleryUiProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
    });
  const trigger = () => host.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]');

  beforeEach(() => {
    onChange.mockReset();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('offers no role until a clip is set, then shows the one the clip is in', async () => {
    await mount(null);
    expect(trigger()).toBeNull();

    await mount(CLIP);
    expect(trigger()?.textContent).toContain('Its soundtrack');

    await mount({ ...CLIP, role: 'video' });
    expect(trigger()?.textContent).toContain('Its picture');
  });

  it('says what an empty slot is for, and why a blocked one cannot be used', async () => {
    await mount(null);
    expect(host.textContent).toContain('Drop a clip to generate the other half of it.');

    // A disabled slot has to explain itself: the reason is the only thing telling the user which
    // of four other controls to clear.
    await mount(null, { reason: 'Clear the First Frame to use a conditioning clip.' });
    expect(host.textContent).toContain('Clear the First Frame to use a conditioning clip.');
  });

  it('warns that a held picture comes back re-encoded, and only in that role', async () => {
    await mount({ ...CLIP, role: 'video' });
    expect(host.textContent).toContain('The clip is re-encoded at the canvas.');

    await mount(CLIP);
    expect(host.textContent).not.toContain('The clip is re-encoded at the canvas.');
  });

  it('keeps the role control out of reach while the slot is blocked', async () => {
    await mount(CLIP, { reason: 'blocked' });
    expect(trigger()?.disabled).toBe(true);
  });

  it('emits the picked role against the same clip', async () => {
    await mount(CLIP);
    await act(() => trigger()!.click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(2);

    const picture = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
      option.textContent?.includes('Its picture')
    );

    await act(() => picture!.click());

    expect(onChange).toHaveBeenCalledWith({ ...CLIP, role: 'video' });
  });
});
