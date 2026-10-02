import type { VideoSourceClip } from '@features/video/core/types';

import { galleryItems } from '@features/gallery';
import { videoClipSpanSeconds } from '@features/video/core/settings';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { IconButton } from '@platform/ui/Button';
import { PauseIcon, PlayIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import { useVideoUiActions } from './VideoUiContext';

/**
 * Controls only its own playback token. Every play requests the current trim from its start; pausing leaves the
 * loop armed for native resume. Playback remains available while editing is disabled.
 */

export const PlayClipSpanButton = memo(function PlayClipSpanButton({ clip }: { clip: VideoSourceClip }) {
  const { t } = useTranslation();
  const { playVideoSpanInPreview, reportError, videoSpanPlayback } = useVideoUiActions();
  const [isResolving, setIsResolving] = useState(false);
  // The token of this button's last request; what the player's report is matched against.
  const [requestToken, setRequestToken] = useState<number | null>(null);
  const playback = useSyncExternalStore(
    videoSpanPlayback.subscribe,
    videoSpanPlayback.getState,
    videoSpanPlayback.getState
  );
  const isPlaying = playback !== null && playback.token === requestToken && playback.isPlaying;
  const span = useMemo(() => videoClipSpanSeconds(clip), [clip]);
  const videoName = clip.video_name;
  const handlePress = useCallback(() => {
    // Use aria-disabled plus this guard to retain keyboard focus during lookup.
    if (!span || isResolving) {
      return;
    }

    if (isPlaying) {
      playback.pause();
      return;
    }

    const owner = captureAccountScope();

    setIsResolving(true);
    galleryItems
      .resolve({ kind: 'video', name: videoName }, owner.signal)
      .then((item) => {
        if (item.kind === 'video' && isAccountScopeCurrent(owner)) {
          const token = playVideoSpanInPreview({ ...span, item });

          // Preview refusal leaves the previous loop armed under this button's token.
          if (token !== null) {
            setRequestToken(token);
          }
        }
      })
      .catch((error: unknown) => {
        if (isAccountScopeCurrent(owner)) {
          reportError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => setIsResolving(false));
  }, [isPlaying, isResolving, playVideoSpanInPreview, playback, reportError, span, videoName]);

  if (!span) {
    return null;
  }

  const label = isPlaying ? t('widgets.video.pauseSelection') : t('widgets.video.playSelection');

  return (
    <IconButton
      aria-busy={isResolving}
      aria-disabled={isResolving}
      aria-label={label}
      opacity={isResolving ? 0.5 : undefined}
      size="2xs"
      title={label}
      variant="ghost"
      onClick={handlePress}
    >
      {isPlaying ? <PauseIcon size={12} /> : <PlayIcon size={12} />}
    </IconButton>
  );
});
