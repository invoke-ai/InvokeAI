import type { ComboboxOption } from '@invoke-ai/ui-library';
import {
  Box,
  Checkbox,
  Combobox,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  IconButton,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Spinner,
  Text,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { SingleValue } from 'chakra-react-select';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { ChangeEvent, UIEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiMagnifyingGlassBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import type { ImageSearchArgs, StarredMode } from 'services/api/endpoints/imageSearch';
import { useSearchImagesQuery } from 'services/api/endpoints/imageSearch';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';
import { useDebounce } from 'use-debounce';

import { GalleryImage } from './ImageGrid/GalleryImage';

const PAGE_SIZE = 60;
const LOAD_MORE_THRESHOLD_PX = 300;
const SEARCH_DEBOUNCE_MS = 300;

type DimensionMode = 'lte' | 'eq' | 'gte' | 'between';

type EffectiveSearchFormState = Omit<ImageSearchArgs, 'offset' | 'limit'>;

type SearchFormState = EffectiveSearchFormState & {
  width_mode: DimensionMode;
  height_mode: DimensionMode;
};

type AnchorSnapshot = {
  imageName: string;
  topInContainer: number;
};

const defaultFormState: SearchFormState = {
  file_name_enabled: true,
  file_name_term: '',
  metadata_enabled: true,
  metadata_term: '',
  width_enabled: false,
  width_mode: 'between',
  width_min: '',
  width_max: '',
  width_exact: '',
  height_enabled: false,
  height_mode: 'between',
  height_min: '',
  height_max: '',
  height_exact: '',
  board_ids: [],
  starred_mode: 'include',
};

const SearchResultItem = memo(({ imageDTO }: { imageDTO: ImageDTO }) => <GalleryImage imageDTO={imageDTO} />);
SearchResultItem.displayName = 'SearchResultItem';

export const ImageSearchModal = memo(() => {
  const [isOpen, setIsOpen] = useState(false);
  const [formState, setFormState] = useState<SearchFormState>(defaultFormState);
  const [page, setPage] = useState(0);
  const [allItems, setAllItems] = useState<ImageDTO[]>([]);
  const [total, setTotal] = useState(0);

  const resultsRef = useRef<HTMLDivElement | null>(null);

  const { data: boards } = useListAllBoardsQuery({ include_archived: true });

  const openModal = useCallback(() => setIsOpen(true), []);
  const closeModal = useCallback(() => setIsOpen(false), []);

  useRegisteredHotkeys({
    category: 'gallery',
    id: 'openSearch',
    callback: (e) => {
      e.preventDefault();
      openModal();
    },
    options: { enabled: true, preventDefault: true },
    dependencies: [openModal],
  });

  const dimensionModeOptions = useMemo<ComboboxOption[]>(
    () => [
      { value: 'lte', label: '≤' },
      { value: 'eq', label: '=' },
      { value: 'gte', label: '≥' },
      { value: 'between', label: 'Between' },
    ],
    []
  );

  const starredModeOptions = useMemo<ComboboxOption[]>(
    () => [
      { value: 'include', label: 'Include Starred' },
      { value: 'exclude', label: 'Exclude Starred' },
      { value: 'only', label: 'Only Starred' },
    ],
    []
  );

  const widthModeValue = useMemo(
    () => dimensionModeOptions.find((opt) => opt.value === formState.width_mode),
    [dimensionModeOptions, formState.width_mode]
  );

  const heightModeValue = useMemo(
    () => dimensionModeOptions.find((opt) => opt.value === formState.height_mode),
    [dimensionModeOptions, formState.height_mode]
  );

  const starredModeValue = useMemo(
    () => starredModeOptions.find((opt) => opt.value === formState.starred_mode),
    [formState.starred_mode, starredModeOptions]
  );

  const onStarredModeChange = useCallback((v: SingleValue<ComboboxOption>) => {
    assert(v?.value === 'include' || v?.value === 'exclude' || v?.value === 'only');
    setFormState((p) => ({ ...p, starred_mode: v.value as StarredMode }));
  }, []);

  const getAnchorSnapshot = useCallback((): AnchorSnapshot | null => {
    const container = resultsRef.current;
    if (!container) {
      return null;
    }
    const containerRect = container.getBoundingClientRect();
    const itemEls = container.querySelectorAll<HTMLElement>('[data-image-name]');
    for (const el of itemEls) {
      const rect = el.getBoundingClientRect();
      if (rect.bottom > containerRect.top) {
        const imageName = el.dataset.imageName;
        if (!imageName) {
          continue;
        }
        return {
          imageName,
          topInContainer: rect.top - containerRect.top,
        };
      }
    }
    return null;
  }, []);

  const restoreAnchorSnapshot = useCallback((anchor: AnchorSnapshot | null) => {
    if (!anchor) {
      return;
    }
    const container = resultsRef.current;
    if (!container) {
      return;
    }
    window.requestAnimationFrame(() => {
      const target = container.querySelector<HTMLElement>(`[data-image-name="${anchor.imageName}"]`);
      if (!target) {
        return;
      }
      const containerRect = container.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const currentTop = targetRect.top - containerRect.top;
      container.scrollTop += currentTop - anchor.topInContainer;
    });
  }, []);

  const onFileNameEnabledChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, file_name_enabled: e.target.checked }));
  }, []);
  const onFileNameTermChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, file_name_term: e.target.value }));
  }, []);
  const onMetadataEnabledChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, metadata_enabled: e.target.checked }));
  }, []);
  const onMetadataTermChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, metadata_term: e.target.value }));
  }, []);
  const onWidthEnabledChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, width_enabled: e.target.checked }));
  }, []);
  const onHeightEnabledChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, height_enabled: e.target.checked }));
  }, []);

  const onWidthModeChange = useCallback((v: SingleValue<ComboboxOption>) => {
    assert(v?.value === 'lte' || v?.value === 'eq' || v?.value === 'gte' || v?.value === 'between');
    setFormState((p) => ({ ...p, width_mode: v.value as DimensionMode }));
  }, []);

  const onHeightModeChange = useCallback((v: SingleValue<ComboboxOption>) => {
    assert(v?.value === 'lte' || v?.value === 'eq' || v?.value === 'gte' || v?.value === 'between');
    setFormState((p) => ({ ...p, height_mode: v.value as DimensionMode }));
  }, []);

  const onWidthMinChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, width_min: e.target.value }));
  }, []);
  const onWidthMaxChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, width_max: e.target.value }));
  }, []);
  const onWidthExactChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, width_exact: e.target.value }));
  }, []);
  const onHeightMinChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, height_min: e.target.value }));
  }, []);
  const onHeightMaxChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, height_max: e.target.value }));
  }, []);
  const onHeightExactChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({ ...p, height_exact: e.target.value }));
  }, []);

  const onAllBoardsChange = useCallback(() => {
    setFormState((p) => ({ ...p, board_ids: [] }));
  }, []);

  const onUncategorizedChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setFormState((p) => ({
      ...p,
      board_ids: e.target.checked ? [...new Set([...p.board_ids, 'none'])] : p.board_ids.filter((id) => id !== 'none'),
    }));
  }, []);

  const onBoardToggle = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const boardId = e.target.value;
    setFormState((p) => ({
      ...p,
      board_ids: e.target.checked
        ? [...new Set([...p.board_ids, boardId])]
        : p.board_ids.filter((id) => id !== boardId),
    }));
  }, []);

  const effectiveFileNameTerm = formState.file_name_enabled ? formState.file_name_term : '';
  const effectiveMetadataTerm = formState.metadata_enabled ? formState.metadata_term : '';
  const effectiveWidthMin =
    formState.width_enabled && (formState.width_mode === 'gte' || formState.width_mode === 'between')
      ? formState.width_min
      : '';
  const effectiveWidthMax =
    formState.width_enabled && (formState.width_mode === 'lte' || formState.width_mode === 'between')
      ? formState.width_max
      : '';
  const effectiveWidthExact = formState.width_enabled && formState.width_mode === 'eq' ? formState.width_exact : '';
  const effectiveHeightMin =
    formState.height_enabled && (formState.height_mode === 'gte' || formState.height_mode === 'between')
      ? formState.height_min
      : '';
  const effectiveHeightMax =
    formState.height_enabled && (formState.height_mode === 'lte' || formState.height_mode === 'between')
      ? formState.height_max
      : '';
  const effectiveHeightExact = formState.height_enabled && formState.height_mode === 'eq' ? formState.height_exact : '';

  const effectiveFormState = useMemo(() => {
    const boardIds = formState.board_ids.length > 0 ? formState.board_ids : EMPTY_ARRAY;

    return {
      board_ids: boardIds,
      starred_mode: formState.starred_mode,
      file_name_enabled: formState.file_name_enabled,
      file_name_term: effectiveFileNameTerm,
      metadata_enabled: formState.metadata_enabled,
      metadata_term: effectiveMetadataTerm,
      width_enabled: formState.width_enabled,
      width_min: effectiveWidthMin,
      width_max: effectiveWidthMax,
      width_exact: effectiveWidthExact,
      height_enabled: formState.height_enabled,
      height_min: effectiveHeightMin,
      height_max: effectiveHeightMax,
      height_exact: effectiveHeightExact,
    } satisfies EffectiveSearchFormState;
  }, [
    effectiveFileNameTerm,
    effectiveHeightExact,
    effectiveHeightMax,
    effectiveHeightMin,
    effectiveMetadataTerm,
    effectiveWidthExact,
    effectiveWidthMax,
    effectiveWidthMin,
    formState.board_ids,
    formState.file_name_enabled,
    formState.height_enabled,
    formState.metadata_enabled,
    formState.starred_mode,
    formState.width_enabled,
  ]);

  const [debouncedEffectiveFormState] = useDebounce(effectiveFormState, SEARCH_DEBOUNCE_MS);

  const queryArgs = useMemo<ImageSearchArgs>(
    () => ({
      ...debouncedEffectiveFormState,
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
    }),
    [debouncedEffectiveFormState, page]
  );

  const headQueryArgs = useMemo<ImageSearchArgs>(
    () => ({
      ...debouncedEffectiveFormState,
      offset: 0,
      limit: PAGE_SIZE,
    }),
    [debouncedEffectiveFormState]
  );

  const { data, isFetching } = useSearchImagesQuery(queryArgs, { skip: !isOpen });
  const { data: headData } = useSearchImagesQuery(headQueryArgs, { skip: !isOpen || page === 0 });

  useEffect(() => {
    setPage(0);
    setAllItems([]);
  }, [debouncedEffectiveFormState]);

  useEffect(() => {
    if (!data) {
      return;
    }

    setTotal(data.total);
    setAllItems((prev) => {
      if (page === 0) {
        return data.items;
      }
      const seen = new Set(prev.map((i) => i.image_name));
      const next = [...prev];
      for (const item of data.items) {
        if (!seen.has(item.image_name)) {
          next.push(item);
        }
      }
      return next;
    });
  }, [data, page]);

  useEffect(() => {
    if (!headData || page === 0) {
      return;
    }
    const anchor = getAnchorSnapshot();
    setTotal(headData.total);
    setAllItems((prev) => {
      const pinned = [...headData.items];
      const pinnedNames = new Set(pinned.map((i) => i.image_name));
      for (const item of prev) {
        if (!pinnedNames.has(item.image_name)) {
          pinned.push(item);
        }
      }
      return pinned;
    });
    restoreAnchorSnapshot(anchor);
  }, [getAnchorSnapshot, headData, page, restoreAnchorSnapshot]);

  const hasMore = allItems.length < total;

  const onResultsScroll = useCallback(
    (e: UIEvent<HTMLDivElement>) => {
      const target = e.currentTarget;

      if (isFetching || !hasMore) {
        return;
      }

      const pixelsFromBottom = target.scrollHeight - target.scrollTop - target.clientHeight;
      if (pixelsFromBottom < LOAD_MORE_THRESHOLD_PX) {
        setPage((p) => p + 1);
      }
    },
    [hasMore, isFetching]
  );

  return (
    <>
      <IconButton
        size="sm"
        variant="link"
        aria-label="Open search"
        icon={<PiMagnifyingGlassBold />}
        onClick={openModal}
      />
      <Modal isOpen={isOpen} onClose={closeModal} size="2xl" isCentered useInert={false}>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="68rem">
          <ModalHeader>Search</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDirection="column" gap={3} minH={0}>
            <Flex direction="column" gap={2}>
              <Flex gap={2} alignItems="center">
                <Checkbox isChecked={formState.file_name_enabled} onChange={onFileNameEnabledChange}>
                  File Name
                </Checkbox>
                <Input value={formState.file_name_term} onChange={onFileNameTermChange} placeholder="contains..." />
              </Flex>
              <Flex gap={2} alignItems="center">
                <Checkbox isChecked={formState.metadata_enabled} onChange={onMetadataEnabledChange}>
                  Metadata
                </Checkbox>
                <Input value={formState.metadata_term} onChange={onMetadataTermChange} placeholder="contains..." />
              </Flex>
              <Flex gap={2} alignItems="center">
                <Checkbox isChecked={formState.width_enabled} onChange={onWidthEnabledChange}>
                  Width
                </Checkbox>
                <Box minW="10rem">
                  <Combobox
                    isDisabled={!formState.width_enabled}
                    isSearchable={false}
                    value={widthModeValue}
                    options={dimensionModeOptions}
                    onChange={onWidthModeChange}
                  />
                </Box>
                {formState.width_mode === 'between' ? (
                  <>
                    <Input
                      placeholder="Min"
                      isDisabled={!formState.width_enabled}
                      value={formState.width_min}
                      onChange={onWidthMinChange}
                    />
                    <Input
                      placeholder="Max"
                      isDisabled={!formState.width_enabled}
                      value={formState.width_max}
                      onChange={onWidthMaxChange}
                    />
                  </>
                ) : null}
                {formState.width_mode === 'gte' ? (
                  <Input
                    placeholder="Min"
                    isDisabled={!formState.width_enabled}
                    value={formState.width_min}
                    onChange={onWidthMinChange}
                  />
                ) : null}
                {formState.width_mode === 'lte' ? (
                  <Input
                    placeholder="Max"
                    isDisabled={!formState.width_enabled}
                    value={formState.width_max}
                    onChange={onWidthMaxChange}
                  />
                ) : null}
                {formState.width_mode === 'eq' ? (
                  <Input
                    placeholder="Exact"
                    isDisabled={!formState.width_enabled}
                    value={formState.width_exact}
                    onChange={onWidthExactChange}
                  />
                ) : null}
              </Flex>
              <Flex gap={2} alignItems="center">
                <Checkbox isChecked={formState.height_enabled} onChange={onHeightEnabledChange}>
                  Height
                </Checkbox>
                <Box minW="10rem">
                  <Combobox
                    isDisabled={!formState.height_enabled}
                    isSearchable={false}
                    value={heightModeValue}
                    options={dimensionModeOptions}
                    onChange={onHeightModeChange}
                  />
                </Box>
                {formState.height_mode === 'between' ? (
                  <>
                    <Input
                      placeholder="Min"
                      isDisabled={!formState.height_enabled}
                      value={formState.height_min}
                      onChange={onHeightMinChange}
                    />
                    <Input
                      placeholder="Max"
                      isDisabled={!formState.height_enabled}
                      value={formState.height_max}
                      onChange={onHeightMaxChange}
                    />
                  </>
                ) : null}
                {formState.height_mode === 'gte' ? (
                  <Input
                    placeholder="Min"
                    isDisabled={!formState.height_enabled}
                    value={formState.height_min}
                    onChange={onHeightMinChange}
                  />
                ) : null}
                {formState.height_mode === 'lte' ? (
                  <Input
                    placeholder="Max"
                    isDisabled={!formState.height_enabled}
                    value={formState.height_max}
                    onChange={onHeightMaxChange}
                  />
                ) : null}
                {formState.height_mode === 'eq' ? (
                  <Input
                    placeholder="Exact"
                    isDisabled={!formState.height_enabled}
                    value={formState.height_exact}
                    onChange={onHeightExactChange}
                  />
                ) : null}
              </Flex>
              <FormControl>
                <FormLabel>Starred</FormLabel>
                <Combobox
                  isSearchable={false}
                  value={starredModeValue}
                  options={starredModeOptions}
                  onChange={onStarredModeChange}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Boards</FormLabel>
                <Flex direction="column" maxH="10rem" overflowY="auto" borderWidth="1px" borderRadius="base" p={2}>
                  <Checkbox isChecked={formState.board_ids.length === 0} onChange={onAllBoardsChange}>
                    All Boards
                  </Checkbox>
                  <Checkbox isChecked={formState.board_ids.includes('none')} onChange={onUncategorizedChange}>
                    Uncategorized
                  </Checkbox>
                  {boards?.map((b) => (
                    <Checkbox
                      key={b.board_id}
                      isChecked={formState.board_ids.includes(b.board_id)}
                      value={b.board_id}
                      onChange={onBoardToggle}
                    >
                      {b.board_name}
                    </Checkbox>
                  ))}
                </Flex>
              </FormControl>
            </Flex>
            <Divider />
            <Box ref={resultsRef} flexGrow={1} minH={0} overflowY="auto" onScroll={onResultsScroll}>
              <Flex direction="column" gap={2} pb={8}>
                <Text fontSize="sm" color="base.500">
                  {isFetching && allItems.length === 0 ? 'Searching…' : `${allItems.length} / ${total} images`}
                </Text>
                <Grid templateColumns="repeat(auto-fill, minmax(9rem, 1fr))" gap={2}>
                  {allItems.map((imageDTO) => (
                    <Box key={imageDTO.image_name} data-image-name={imageDTO.image_name}>
                      <SearchResultItem imageDTO={imageDTO} />
                    </Box>
                  ))}
                </Grid>
                {isFetching && allItems.length > 0 ? (
                  <Flex justifyContent="center" py={2}>
                    <Spinner size="sm" />
                  </Flex>
                ) : null}
              </Flex>
            </Box>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
});

ImageSearchModal.displayName = 'ImageSearchModal';
