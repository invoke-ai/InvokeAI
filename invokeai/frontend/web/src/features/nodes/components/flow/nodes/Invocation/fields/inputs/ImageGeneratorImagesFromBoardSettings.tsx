import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { ImageGeneratorImagesFromBoard } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

type ImageGeneratorImagesFromBoardSettingsProps = {
  state: ImageGeneratorImagesFromBoard;
  onChange: (state: ImageGeneratorImagesFromBoard) => void;
};
export const ImageGeneratorImagesFromBoardSettings = memo(
  ({ state, onChange }: ImageGeneratorImagesFromBoardSettingsProps) => {
    const { t } = useTranslation();

    const onChangeCategory = useCallback(
      (category: 'images' | 'assets') => {
        onChange({ ...state, category });
      },
      [onChange, state]
    );
    const onChangeBoardId = useCallback(
      (board_id: string) => {
        onChange({ ...state, board_id });
      },
      [onChange, state]
    );

    return (
      <Flex gap={2} flexDir="column">
        <FormControl orientation="vertical">
          <FormLabel>{t('common.board')}</FormLabel>
          <BoardCombobox board_id={state.board_id} onChange={onChangeBoardId} />
        </FormControl>
        <FormControl orientation="vertical">
          <FormLabel>{t('nodes.generatorImagesCategory')}</FormLabel>
          <CategoryCombobox category={state.category} onChange={onChangeCategory} />
        </FormControl>
      </Flex>
    );
  }
);
ImageGeneratorImagesFromBoardSettings.displayName = 'ImageGeneratorImagesFromBoardSettings';

const listAllBoardsQueryArg = { include_archived: false };

const BoardCombobox = ({
  board_id,
  onChange: _onChange,
}: {
  board_id: string | undefined;
  onChange: (board_id: string) => void;
}) => {
  const { t } = useTranslation();
  const listAllBoardsQuery = useListAllBoardsQuery(listAllBoardsQueryArg);

  const options = useMemo<ComboboxOption[]>(() => {
    if (!listAllBoardsQuery.data) {
      return EMPTY_ARRAY;
    }
    return listAllBoardsQuery.data.map((board) => ({
      label: board.board_name,
      value: board.board_id,
    }));
  }, [listAllBoardsQuery.data]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        // This should never happen
        return;
      }

      _onChange(v.value);
    },
    [_onChange]
  );

  const value = useMemo(() => options.find((o) => o.value === board_id) ?? null, [board_id, options]);

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  return (
    <Combobox
      className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
      value={value}
      options={options}
      onChange={onChange}
      placeholder={t('boards.selectBoard')}
      noOptionsMessage={noOptionsMessage}
    />
  );
};

const CategoryCombobox = ({
  category,
  onChange: _onChange,
}: {
  category: 'images' | 'assets';
  onChange: (category: 'images' | 'assets') => void;
}) => {
  const { t } = useTranslation();

  const options = useMemo<ComboboxOption[]>(
    () => [
      { label: t('gallery.images'), value: 'images' },
      { label: t('gallery.assets'), value: 'assets' },
    ],
    [t]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v || (v.value !== 'images' && v.value !== 'assets')) {
        // This should never happen
        return;
      }

      _onChange(v.value);
    },
    [_onChange]
  );

  const value = useMemo(() => options.find((o) => o.value === category), [options, category]);

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  return (
    <Combobox
      className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
      value={value}
      options={options}
      onChange={onChange}
      placeholder={t('boards.selectBoard')}
      noOptionsMessage={noOptionsMessage}
    />
  );
};
