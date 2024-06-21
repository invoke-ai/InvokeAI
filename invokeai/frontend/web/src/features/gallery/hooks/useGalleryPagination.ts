import { useMemo, useCallback } from "react";
import { useAppDispatch, useAppSelector } from "../../../app/store/storeHooks";
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from "../../../services/api/endpoints/boards";
import { useListImagesQuery } from "../../../services/api/endpoints/images";
import { selectListImagesQueryArgs } from "../store/gallerySelectors";
import { offsetChanged } from "../store/gallerySlice";

export const useGalleryPagination = () => {
    const dispatch = useAppDispatch();
    const { offset, limit } = useAppSelector((s) => s.gallery);
    const queryArgs = useAppSelector(selectListImagesQueryArgs);

    const { count, total } = useListImagesQuery(queryArgs, {
        selectFromResult: ({ data }) => ({ count: data?.items.length ?? 0, total: data?.total ?? 0 }),
    });

    const currentPage = useMemo(() => Math.ceil(offset / (limit || 0)), [offset, limit]);
    const pages = useMemo(() => Math.ceil(total / (limit || 0)), [total, limit]);

    const isNextEnabled = useMemo(() => {
        if (!count) {
            return false;
        }
        return currentPage + 1 < pages;
    }, [count, currentPage, pages]);
    const isPrevEnabled = useMemo(() => {
        if (!count) {
            return false;
        }
        return offset > 0;
    }, [count, offset]);

    const goNext = useCallback(() => {
        dispatch(offsetChanged(offset + (limit || 0)));
    }, [dispatch, offset, limit]);

    const goPrev = useCallback(() => {
        dispatch(offsetChanged(Math.max(offset - (limit || 0), 0)));
    }, [dispatch, offset, limit]);

    const goToPage = useCallback(
        (page: number) => {
            const p = Math.max(0, Math.min(page, pages - 1));
            dispatch(offsetChanged(page * (limit || 0)));
        },
        [dispatch, pages, limit]
    );
    const goToFirst = useCallback(() => {
        dispatch(offsetChanged(0));
    }, [dispatch]);
    const goToLast = useCallback(() => {
        dispatch(offsetChanged(pages * (limit || 0)));
    }, [dispatch, pages, limit]);

    // calculate the page buttons to display - current page with 3 around it
    const pageButtons = useMemo(() => {
        const buttons = [];
        const maxPageButtons = 3;
        let startPage = Math.max(currentPage - (Math.floor(maxPageButtons / 2)), 0);
        let endPage = Math.min(startPage + maxPageButtons - 1, pages - 1);

        if (endPage - startPage < maxPageButtons - 1) {
            startPage = Math.max(endPage - maxPageButtons + 1, 0);
        }

        if (startPage > 0) {
            buttons.push(0);
            if (startPage > 1) {
                buttons.push('...');
            }
        }

        for (let i = startPage; i <= endPage; i++) {
            buttons.push(i);
        }

        if (endPage < pages - 1) {
            if (endPage < pages - 2) {
                buttons.push('...');
            }
            buttons.push(pages - 1);
        }

        return buttons;
    }, [currentPage, pages]);

    const isFirstEnabled = useMemo(() => currentPage > 0, [currentPage]);
    const isLastEnabled = useMemo(() => currentPage < pages - 1, [currentPage, pages]);

    const rangeDisplay = useMemo(() => {
        const startItem = currentPage * (limit || 0) + 1;
        const endItem = Math.min((currentPage + 1) * (limit || 0), total);
        return `${startItem}-${endItem} of ${total}`;
    }, [total, currentPage, limit]);

    const api = useMemo(
        () => ({
            count,
            total,
            currentPage,
            pages,
            isNextEnabled,
            isPrevEnabled,
            goNext,
            goPrev,
            goToPage,
            goToFirst,
            goToLast,
            pageButtons,
            isFirstEnabled,
            isLastEnabled,
            rangeDisplay
        }),
        [
            count,
            total,
            currentPage,
            pages,
            isNextEnabled,
            isPrevEnabled,
            goNext,
            goPrev,
            goToPage,
            goToFirst,
            goToLast,
            pageButtons,
            isFirstEnabled,
            isLastEnabled,
            rangeDisplay
        ]
    );
    return api;
};