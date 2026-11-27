import { Button, ButtonGroup, Divider, Flex, Grid, GridItem, IconButton, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import { getDefaultStringOption, type NodeFieldStringDropdownSettings } from 'features/nodes/types/workflow';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiPlusBold, PiXBold } from 'react-icons/pi';

const overlayscrollbarsOptions = getOverlayScrollbarsParams({}).options;

export const NodeFieldElementStringDropdownSettings = memo(
  ({ id, settings }: { id: string; settings: NodeFieldStringDropdownSettings }) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onRemoveOption = useCallback(
      (index: number) => {
        const options = [...settings.options];
        options.splice(index, 1);
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: { ...settings, options } } }));
      },
      [settings, dispatch, id]
    );

    const onChangeOptionValue = useCallback(
      (index: number, value: string) => {
        if (!settings.options[index]) {
          return;
        }
        const option = { ...settings.options[index], value };
        const options = [...settings.options];
        options[index] = option;
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: { ...settings, options } } }));
      },
      [dispatch, id, settings]
    );

    const onChangeOptionLabel = useCallback(
      (index: number, label: string) => {
        if (!settings.options[index]) {
          return;
        }
        const option = { ...settings.options[index], label };
        const options = [...settings.options];
        options[index] = option;
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: { ...settings, options } } }));
      },
      [dispatch, id, settings]
    );

    const onAddOption = useCallback(() => {
      const options = [...settings.options, getDefaultStringOption()];
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: { ...settings, options } } }));
    }, [dispatch, id, settings]);

    const onResetOptions = useCallback(() => {
      const options = [getDefaultStringOption()];
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: { ...settings, options } } }));
    }, [dispatch, id, settings]);

    return (
      <Flex
        className={NO_DRAG_CLASS}
        position="relative"
        w="full"
        h="auto"
        maxH={64}
        alignItems="stretch"
        justifyContent="center"
        borderRadius="base"
        flexDir="column"
        gap={2}
      >
        <ButtonGroup isAttached={false} w="full">
          <Button onClick={onAddOption} variant="ghost" flex={1} leftIcon={<PiPlusBold />}>
            {t('workflows.builder.addOption')}
          </Button>
          <Button onClick={onResetOptions} variant="ghost" flex={1} leftIcon={<PiArrowCounterClockwiseBold />}>
            {t('workflows.builder.resetOptions')}
          </Button>
        </ButtonGroup>
        {settings.options.length > 0 && (
          <>
            <Divider />
            <OverlayScrollbarsComponent
              className={NO_WHEEL_CLASS}
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid gap={1} gridTemplateColumns="auto 1fr 1fr auto" gridTemplateRows="auto 1fr" alignItems="center">
                <GridItem minW={8}>
                  <Text textAlign="center" variant="subtext">
                    #
                  </Text>
                </GridItem>
                <GridItem>
                  <Text textAlign="center" variant="subtext">
                    {t('common.label')}
                  </Text>
                </GridItem>
                <GridItem>
                  <Text textAlign="center" variant="subtext">
                    {t('common.value')}
                  </Text>
                </GridItem>
                <GridItem />
                {settings.options.map(({ value, label }, index) => (
                  <ListItemContent
                    key={index}
                    value={value}
                    label={label}
                    index={index}
                    onRemoveOption={onRemoveOption}
                    onChangeOptionValue={onChangeOptionValue}
                    onChangeOptionLabel={onChangeOptionLabel}
                    isRemoveDisabled={settings.options.length <= 1}
                  />
                ))}
              </Grid>
            </OverlayScrollbarsComponent>
          </>
        )}
      </Flex>
    );
  }
);

NodeFieldElementStringDropdownSettings.displayName = 'NodeFieldElementStringDropdownSettings';

type ListItemContentProps = {
  value: string;
  label: string;
  index: number;
  onRemoveOption: (index: number) => void;
  onChangeOptionValue: (index: number, value: string) => void;
  onChangeOptionLabel: (index: number, label: string) => void;
  isRemoveDisabled: boolean;
};

const ListItemContent = memo(
  ({
    value,
    label,
    index,
    onRemoveOption,
    onChangeOptionValue,
    onChangeOptionLabel,
    isRemoveDisabled,
  }: ListItemContentProps) => {
    const { t } = useTranslation();

    const onClickRemove = useCallback(() => {
      onRemoveOption(index);
    }, [index, onRemoveOption]);

    const onChangeValue = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        onChangeOptionValue(index, e.target.value);
      },
      [index, onChangeOptionValue]
    );

    const onChangeLabel = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        onChangeOptionLabel(index, e.target.value);
      },
      [index, onChangeOptionLabel]
    );

    return (
      <>
        <GridItem>
          <Text variant="subtext" textAlign="center">
            {index + 1}.
          </Text>
        </GridItem>
        <GridItem>
          <Input size="sm" resize="none" placeholder="label" value={label} onChange={onChangeLabel} />
        </GridItem>
        <GridItem>
          <Input size="sm" resize="none" placeholder="value" value={value} onChange={onChangeValue} />
        </GridItem>
        <GridItem>
          <IconButton
            tabIndex={-1}
            size="sm"
            variant="link"
            minW={8}
            minH={8}
            onClick={onClickRemove}
            isDisabled={isRemoveDisabled}
            icon={<PiXBold />}
            aria-label={t('common.delete')}
          />
        </GridItem>
      </>
    );
  }
);
ListItemContent.displayName = 'ListItemContent';
