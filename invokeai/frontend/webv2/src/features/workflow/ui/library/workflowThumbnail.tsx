import type { WorkflowThumbnailFit } from '@features/workflow/ui/WorkflowUiContext';

import { Image } from '@chakra-ui/react';
import { createContext, use } from 'react';

/** Every workflow thumbnail well is 3:2, whatever the image inside it. */
export const WORKFLOW_THUMBNAIL_ASPECT_RATIO = 3 / 2;

/**
 * The account's Fill/Fit choice for every thumbnail under the library dialog. A context rather than a store read per
 * card: one subscription in the dialog, and surfaces mounted without it show the default fill.
 */
export const WorkflowThumbnailFitContext = createContext<WorkflowThumbnailFit>('cover');

/** A thumbnail filling its well, cropped (`cover`) or letterboxed (`contain`) as the account chose. */
export const WorkflowThumbnailImage = ({ onError, src }: { onError: () => void; src: string | null | undefined }) => {
  const fit = use(WorkflowThumbnailFitContext);

  return <Image alt="" h="full" objectFit={fit} src={src ?? undefined} w="full" onError={onError} />;
};
