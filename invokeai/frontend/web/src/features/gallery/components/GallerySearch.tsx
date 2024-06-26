import { Flex, IconButton, Input, InputGroup, InputRightElement, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { searchTermChanged } from 'features/gallery/store/gallerySlice';
import { motion } from 'framer-motion';
import { debounce } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassBold, PiXBold } from 'react-icons/pi';

export const GallerySearch = () => {
  const dispatch = useAppDispatch();
  const { searchTerm } = useAppSelector((s) => s.gallery);
  const { t } = useTranslation();

  const [expanded, setExpanded] = useState(false);
  const [searchTermInput, setSearchTermInput] = useState('');

  const debouncedSetSearchTerm = useMemo(() => {
    return debounce((value: string) => {
      dispatch(searchTermChanged(value));
    }, 1000);
  }, [dispatch]);

  const onChangeInput = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setSearchTermInput(e.target.value);
      debouncedSetSearchTerm(e.target.value);
    },
    [debouncedSetSearchTerm]
  );

  const onClearInput = useCallback(() => {
    setSearchTermInput('');
    debouncedSetSearchTerm('');
  }, [debouncedSetSearchTerm]);

  const toggleExpanded = useCallback((newState: boolean) => {
    setExpanded(newState);
  }, []);

  return (
    <Flex>
      {!expanded && (
        <Tooltip
          label={
            searchTerm && searchTerm.length ? `${t('gallery.searchingBy')} ${searchTerm}` : t('gallery.noActiveSearch')
          }
        >
          <IconButton
            aria-label="Close"
            icon={<PiMagnifyingGlassBold />}
            onClick={toggleExpanded.bind(null, true)}
            variant="outline"
            size="sm"
          />
        </Tooltip>
      )}
      <motion.div
        initial={false}
        animate={{ width: expanded ? '200px' : '0px' }}
        transition={{ duration: 0.3 }}
        style={{ overflow: 'hidden' }}
      >
        <InputGroup size="sm">
          <IconButton
            aria-label="Close"
            icon={<PiMagnifyingGlassBold />}
            onClick={toggleExpanded.bind(null, false)}
            variant="ghost"
            size="sm"
          />

          <Input
            type="text"
            placeholder="Search..."
            size="sm"
            variant="outline"
            onChange={onChangeInput}
            value={searchTermInput}
          />
          {searchTermInput && searchTermInput.length && (
            <InputRightElement h="full" pe={2}>
              <IconButton
                onClick={onClearInput}
                size="sm"
                variant="link"
                aria-label={t('boards.clearSearch')}
                icon={<PiXBold />}
              />
            </InputRightElement>
          )}
        </InputGroup>
      </motion.div>
    </Flex>
  );
};
