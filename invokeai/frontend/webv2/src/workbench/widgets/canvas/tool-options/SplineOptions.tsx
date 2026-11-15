import type { ToolFormProps, ToolPropertyForm } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { Box, Grid, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { useVectorEditSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useCallback, useMemo, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

const ACTION_ROW_CSS = {
  '& > *': { position: 'relative', minWidth: 0, fontSize: 'xs' },
  '& > * + *::before': {
    content: '""',
    position: 'absolute',
    insetInlineStart: '-1.5px',
    top: '50%',
    transform: 'translateY(-50%)',
    height: '3.5',
    width: '1px',
    bg: 'border.emphasized',
    pointerEvents: 'none',
  },
} as const;

/** Action rows share the Shapes segmented frame without giving commands radio-button semantics. */
const SplineActionRow = ({
  children,
  label,
  columns = 2,
}: {
  children: ReactNode;
  label: string;
  columns?: number;
}) => (
  <Grid
    aria-label={label}
    role="group"
    borderColor="border.subtle"
    borderRadius="control"
    borderWidth="1px"
    gap="0.5"
    p="0.5"
    templateColumns={`repeat(${columns}, minmax(0, 1fr))`}
    css={ACTION_ROW_CSS}
  >
    {children}
  </Grid>
);

const SplineSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const editing = useVectorEditSession(engine);
  const api = engine.layers.vectorEdit;
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const pointTypeOptions = useMemo(
    () =>
      (['corner', 'smooth', 'symmetric'] as const).map((type) => ({
        value: type,
        label: t(`widgets.layers.vector.${type}`),
      })),
    [t]
  );
  const setPointType = useCallback(
    (type: string) => api.setPointType(type as 'corner' | 'smooth' | 'symmetric'),
    [api]
  );
  const smoothPath = useCallback(() => api.smooth(false), [api]);
  const smoothPoints = useCallback(() => api.smooth(true), [api]);
  const selected = !!editing && editing.selectedPoints.length > 0;
  const paths = editing?.paths ?? [];
  const active = paths.find((path) => path.id === editing?.activePathId);
  const pointTypes = new Set(
    editing?.selectedPoints.flatMap((ref) => {
      const point = paths.find((path) => path.id === ref.pathId)?.points[ref.pointIndex];
      return point ? [point.type] : [];
    })
  );
  return (
    <Stack gap="3">
      {!editing ? (
        <Text color="fg.muted" fontSize="sm">
          {label('selectVectorLayerHint')}
        </Text>
      ) : null}
      <SegmentedControl
        ariaLabel={label('pointsTitle')}
        disabled={!selected}
        options={pointTypeOptions}
        value={pointTypes.size === 1 ? ([...pointTypes][0] ?? null) : null}
        onChange={setPointType}
      />
      <SplineActionRow label={label('pointsTitle')}>
        <Button colorPalette="gray" disabled={!editing?.canSplit} size="sm" variant="ghost" onClick={api.split}>
          {label('split')}
        </Button>
        <Button colorPalette="gray" disabled={!editing?.canJoin} size="sm" variant="ghost" onClick={api.join}>
          {label('join')}
        </Button>
      </SplineActionRow>
      <Stack
        borderColor="border.subtle"
        borderTopWidth="1px"
        gap="3"
        mt="2"
        pt="4"
        role="group"
        aria-label={label('pathPropertiesTitle')}
      >
        <Text color="fg.muted" fontWeight="600">
          {label('pathPropertiesTitle')}
        </Text>
        <PropertyControlRow label={label('smooth')}>
          <Box gridColumn="2 / -1" minW="0">
            <SplineActionRow label={label('smooth')}>
              <Button colorPalette="gray" disabled={!active} size="sm" variant="ghost" onClick={smoothPath}>
                {label('pathPropertiesTitle')}
              </Button>
              <Button colorPalette="gray" disabled={!selected} size="sm" variant="ghost" onClick={smoothPoints}>
                {label('pointsTitle')}
              </Button>
            </SplineActionRow>
          </Box>
        </PropertyControlRow>
        <SplineActionRow columns={3} label={label('pathPropertiesTitle')}>
          <Button colorPalette="gray" disabled={!active} size="sm" variant="ghost" onClick={api.copyPath}>
            {label('copyPathButton')}
          </Button>
          <Button colorPalette="gray" disabled={!active} size="sm" variant="ghost" onClick={api.extractPath}>
            {label('extract')}
          </Button>
          <Button colorPalette="gray" disabled={!active} size="sm" variant="ghost" onClick={api.deletePath}>
            {label('deletePathButton')}
          </Button>
        </SplineActionRow>
      </Stack>
    </Stack>
  );
};

export const splineForm: ToolPropertyForm = {
  id: 'spline',
  groups: [{ body: SplineSettings, id: 'spline-points', labelKey: 'widgets.layers.vector.pointsTitle' }],
};
