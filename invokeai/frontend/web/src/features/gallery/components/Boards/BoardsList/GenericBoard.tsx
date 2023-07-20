import { As, Badge, Flex, Box, Icon } from '@chakra-ui/react';
import { TypesafeDroppableData } from 'app/components/ImageDnd/typesafeDnd';
import { BoardId, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { ReactNode, useCallback } from 'react';
import BoardContextMenu from '../BoardContextMenu';
import { useAppDispatch } from '../../../../../app/store/storeHooks';
import { BASE_BADGE_STYLES } from './GalleryBoard';
import { MdFolderOff } from 'react-icons/md';

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
  const { board_id, isSelected, label, badgeCount } = props;

  const dispatch = useAppDispatch();

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected(board_id));
  }, [board_id, dispatch]);

  return (
    <BoardContextMenu board_id={board_id}>
      {(ref) => (
        <Box
          sx={{ w: 'full', h: 'full', touchAction: 'none', userSelect: 'none' }}
        >
          <Flex
            sx={{
              position: 'relative',
              justifyContent: 'center',
              alignItems: 'center',
              aspectRatio: '1/1',
              w: 'full',
              h: 'full',
            }}
          >
            <Flex
              ref={ref}
              onClick={handleSelectBoard}
              sx={{
                w: 'full',
                h: 'full',
                position: 'relative',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: 'base',
                cursor: 'pointer',
              }}
            >
              <Flex
                sx={{
                  w: 'full',
                  h: 'full',
                  justifyContent: 'center',
                  alignItems: 'center',
                  borderRadius: 'base',
                  bg: 'base.200',
                  _dark: {
                    bg: 'base.800',
                  },
                }}
              >
                <Flex
                  sx={{
                    w: 'full',
                    h: 'full',
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <Icon
                    boxSize={12}
                    as={MdFolderOff}
                    sx={{
                      mt: -3,
                      opacity: 0.7,
                      color: 'base.500',
                      _dark: {
                        color: 'base.500',
                      },
                    }}
                  />
                </Flex>
              </Flex>

              <Flex
                sx={{
                  position: 'absolute',
                  insetInlineEnd: 0,
                  top: 0,
                  p: 1,
                }}
              >
                <Badge variant="solid" sx={BASE_BADGE_STYLES}>
                  {badgeCount}
                </Badge>
              </Flex>

              <Box
                className="selection-box"
                sx={{
                  position: 'absolute',
                  top: 0,
                  insetInlineEnd: 0,
                  bottom: 0,
                  insetInlineStart: 0,
                  borderRadius: 'base',
                  transitionProperty: 'common',
                  transitionDuration: 'common',
                  shadow: isSelected ? 'selected.light' : undefined,
                  _dark: {
                    shadow: isSelected ? 'selected.dark' : undefined,
                  },
                }}
              />

              <Flex
                sx={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  p: 1,
                  justifyContent: 'center',
                  alignItems: 'center',
                  w: 'full',
                  maxW: 'full',
                  borderBottomRadius: 'base',
                  bg: isSelected ? 'accent.400' : 'base.500',
                  color: isSelected ? 'base.50' : 'base.100',
                  _dark: {
                    bg: isSelected ? 'accent.500' : 'base.600',
                    color: isSelected ? 'base.50' : 'base.100',
                  },
                  lineHeight: 'short',
                  fontSize: 'xs',
                }}
              >
                {label}
              </Flex>
            </Flex>
          </Flex>
        </Box>
      )}
    </BoardContextMenu>
  );
};

export default GenericBoard;
