import { As, Badge, Flex } from '@chakra-ui/react';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { TypesafeDroppableData } from 'features/dnd/types';
import { BoardId } from 'features/gallery/store/types';
import { ReactNode, memo } from 'react';
import BoardContextMenu from '../BoardContextMenu';

type GenericBoardProps = {
  board_id: BoardId;
  droppableData?: TypesafeDroppableData;
  onClick: () => void;
  isSelected: boolean;
  icon: As;
  label: string;
  dropLabel?: ReactNode;
  badgeCount?: number;
};

export const formatBadgeCount = (count: number) =>
  Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(count);

const GenericBoard = (props: GenericBoardProps) => {
  const {
    board_id,
    droppableData,
    onClick,
    isSelected,
    icon,
    label,
    badgeCount,
    dropLabel,
  } = props;

  return (
    <BoardContextMenu board_id={board_id}>
      {(ref) => (
        <Flex
          ref={ref}
          sx={{
            flexDir: 'column',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
            w: 'full',
            h: 'full',
            borderRadius: 'base',
          }}
        >
          <Flex
            onClick={onClick}
            sx={{
              position: 'relative',
              justifyContent: 'center',
              alignItems: 'center',
              borderRadius: 'base',
              w: 'full',
              aspectRatio: '1/1',
              overflow: 'hidden',
              shadow: isSelected ? 'selected.light' : undefined,
              _dark: { shadow: isSelected ? 'selected.dark' : undefined },
              flexShrink: 0,
            }}
          >
            <IAINoContentFallback
              boxSize={8}
              icon={icon}
              sx={{
                border: '2px solid var(--invokeai-colors-base-200)',
                _dark: { border: '2px solid var(--invokeai-colors-base-800)' },
              }}
            />
            <Flex
              sx={{
                position: 'absolute',
                insetInlineEnd: 0,
                top: 0,
                p: 1,
              }}
            >
              {badgeCount !== undefined && (
                <Badge variant="solid">{formatBadgeCount(badgeCount)}</Badge>
              )}
            </Flex>
            <IAIDroppable data={droppableData} dropLabel={dropLabel} />
          </Flex>
          <Flex
            sx={{
              h: 'full',
              alignItems: 'center',
              fontWeight: isSelected ? 600 : undefined,
              fontSize: 'sm',
              color: isSelected ? 'base.900' : 'base.700',
              _dark: { color: isSelected ? 'base.50' : 'base.200' },
            }}
          >
            {label}
          </Flex>
        </Flex>
      )}
    </BoardContextMenu>
  );
};

export default memo(GenericBoard);
