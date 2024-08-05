import { Button, Flex, FormControl, FormLabel, Input, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { isModalOpenChanged,updatingStylePresetChanged } from 'features/stylePresets/store/slice';
import { toast } from 'features/toast/toast';
import type { ChangeEventHandler} from 'react';
import { useCallback, useEffect, useState } from 'react';
import type {
  StylePresetRecordDTO} from 'services/api/endpoints/stylePresets';
import {
  useCreateStylePresetMutation,
  useUpdateStylePresetMutation,
} from 'services/api/endpoints/stylePresets';

export const StylePresetForm = ({ updatingPreset }: { updatingPreset: StylePresetRecordDTO | null }) => {
  const [createStylePreset] = useCreateStylePresetMutation();
  const [updateStylePreset] = useUpdateStylePresetMutation();
  const dispatch = useAppDispatch();

  const [name, setName] = useState(updatingPreset ? updatingPreset.name : '');
  const [posPrompt, setPosPrompt] = useState(updatingPreset ? updatingPreset.preset_data.positive_prompt : '');
  const [negPrompt, setNegPrompt] = useState(updatingPreset ? updatingPreset.preset_data.negative_prompt : '');

  const handleChangeName = useCallback<ChangeEventHandler<HTMLInputElement>>((e) => {
    setName(e.target.value);
  }, []);

  const handleChangePosPrompt = useCallback<ChangeEventHandler<HTMLTextAreaElement>>((e) => {
    setPosPrompt(e.target.value);
  }, []);

  const handleChangeNegPrompt = useCallback<ChangeEventHandler<HTMLTextAreaElement>>((e) => {
    setNegPrompt(e.target.value);
  }, []);

  useEffect(() => {
    if (updatingPreset) {
      setName(updatingPreset.name);
      setPosPrompt(updatingPreset.preset_data.positive_prompt);
      setNegPrompt(updatingPreset.preset_data.negative_prompt);
    } else {
      setName('');
      setPosPrompt('');
      setNegPrompt('');
    }
  }, [updatingPreset]);

  const handleClickSave = useCallback(async () => {
    try {
      if (updatingPreset) {
        await updateStylePreset({
          id: updatingPreset.id,
          changes: { name, preset_data: { positive_prompt: posPrompt, negative_prompt: negPrompt } },
        }).unwrap();
      } else {
        await createStylePreset({
          name: name,
          preset_data: { positive_prompt: posPrompt, negative_prompt: negPrompt },
        }).unwrap();
      }
    } catch (error) {
      toast({
        status: 'error',
        title: 'Failed to save style preset',
      });
    }

    dispatch(updatingStylePresetChanged(null));
    dispatch(isModalOpenChanged(false));
  }, [dispatch, updatingPreset, name, posPrompt, negPrompt, updateStylePreset, createStylePreset]);

  return (
    <Flex flexDir="column" gap="4">
      <FormControl>
        <FormLabel>Name</FormLabel>
        <Input value={name} onChange={handleChangeName} />
      </FormControl>
      <FormControl>
        <FormLabel>Positive Prompt</FormLabel>
        <Textarea value={posPrompt} onChange={handleChangePosPrompt} />
      </FormControl>
      <FormControl>
        <FormLabel>Negative Prompt</FormLabel>
        <Textarea value={negPrompt} onChange={handleChangeNegPrompt} />
      </FormControl>
      <Button onClick={handleClickSave}>Save</Button>
    </Flex>
  );
};
