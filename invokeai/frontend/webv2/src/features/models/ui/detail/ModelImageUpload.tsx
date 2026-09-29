/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { GalleryItem } from '@features/gallery';
import type { GalleryHost, GalleryPickerAccept } from '@features/gallery/picker';
import type { ModelConfig } from '@features/models/core/types';

import { Box, Image } from '@chakra-ui/react';
import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import { GalleryHostProvider } from '@features/gallery/picker';
import { deleteModelImage, getModelImageUrl, updateModelImage } from '@features/models/data/api';
import { markCoverImageChanged, useModelsSelector } from '@features/models/data/modelsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const ACCEPTED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const IMAGE_ONLY: GalleryPickerAccept = ['image'];
/** The model manager lives on the Launchpad: no project board to seed from and no Gallery widget to reveal. */
const NO_GALLERY_VALUES: Record<string, unknown> = {};

interface ModelImageUploadProps {
  model: Pick<ModelConfig, 'cover_image' | 'key' | 'name'>;
  onError: (message: string) => void;
  onUpdated: () => void;
}

export const ModelImageUpload = (props: ModelImageUploadProps) => (
  <ModelImageUploadForModel key={`${props.model.key}:${props.model.cover_image ?? ''}`} {...props} />
);

/** The cover is chosen like any image slot, as a square tile: from the gallery, where the picker also uploads files. */
const ModelImageUploadForModel = ({ model, onError, onUpdated }: ModelImageUploadProps) => {
  const { t } = useTranslation();
  const imageVersion = useModelsSelector((snapshot) => snapshot.coverImageVersions[model.key]);
  const [hasImage, setHasImage] = useState(Boolean(model.cover_image));
  // Setting and removing share one busy flag: the slot hosts one action at a time.
  const { isBusy, run } = useScopedAction();
  const notify = useNotify();
  const galleryHost = useMemo<GalleryHost>(
    () => ({
      galleryValues: NO_GALLERY_VALUES,
      notifications: {
        add: ({ kind, message, title }) => notify[kind](title, message),
        reportError: ({ message }) => notify.error(message),
      },
      projectName: '',
    }),
    [notify]
  );
  const value = useMemo(
    () => (hasImage ? { kind: 'image' as const, name: t('models.modelCoverAlt', { name: model.name }) } : null),
    [hasImage, model.name, t]
  );
  const thumbnail = useMemo(
    () => (
      <Image
        alt=""
        boxSize="full"
        fit="cover"
        src={getModelImageUrl(model.key, imageVersion ? String(imageVersion) : undefined)}
        onError={() => setHasImage(false)}
      />
    ),
    [imageVersion, model.key]
  );

  // The server keeps the full image as the cover, so take the original rather than the gallery thumbnail.
  const setCover = (item: GalleryItem) => {
    const modelKey = model.key;

    void run(
      async (owner) => {
        const response = await fetch(item.fullUrl, { signal: owner.signal });

        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();

        assertAccountScopeCurrent(owner);
        if (!ACCEPTED_TYPES.has(blob.type)) {
          onError(t('models.useSupportedImage'));
          return;
        }

        await updateModelImage(modelKey, new File([blob], item.name, { type: blob.type }), owner.signal);

        assertAccountScopeCurrent(owner);
        setHasImage(true);
        // Bumps the cache-bust version so this slot and list thumbnails reload.
        markCoverImageChanged(modelKey, true);
        onUpdated();
      },
      (_message, error) => onError(getApiErrorMessage(error, t('models.failedToUploadModelImage')))
    );
  };

  const removeCover = () => {
    const modelKey = model.key;

    void run(
      async (owner) => {
        await deleteModelImage(modelKey, owner.signal);

        assertAccountScopeCurrent(owner);
        setHasImage(false);
        markCoverImageChanged(modelKey, false);
        onUpdated();
      },
      (_message, error) => onError(getApiErrorMessage(error, t('models.failedToRemoveModelImage')))
    );
  };

  return (
    <GalleryHostProvider host={galleryHost}>
      <Box flexShrink={0} w="28">
        <GalleryMediaSlot
          accept={IMAGE_ONLY}
          busy={isBusy}
          dropId={`model-cover:${model.key}`}
          layout="tile"
          thumbnail={thumbnail}
          value={value}
          onChange={(item) => (item ? setCover(item) : removeCover())}
        />
      </Box>
    </GalleryHostProvider>
  );
};
