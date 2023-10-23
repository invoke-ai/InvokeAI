#!/usr/bin/env python

import argparse
import sys

from pathlib import Path
from typing import Optional
from invokeai.backend.normalized_mm.normalized_model_manager import (
    NormalizedModelManager,
    InvalidModelException,
    ModelNotFoundException,
    DuplicateModelException
)
from invokeai.app.services.config import InvokeAIAppConfig

config: InvokeAIAppConfig = InvokeAIAppConfig.get_config()
model_manager: Optional[NormalizedModelManager] = None

def list_parts(args):
    try:
        model = model_manager.get_pipeline(args.model_name)
        print(f"Components of model {args.model_name}:")
        print(f"   {'ROLE':20s} {'TYPE':20s} {'REFCOUNT':8} PATH")
        for role, part in model.parts.items():
            print(f"   {role:20s} {part.type:20s} {part.refcount:4d}     {part.path}")
    except ModelNotFoundException:
        print(f'{args.model_name}: model not found')


def list_models(args):
    model_list = model_manager.list_models()
    print(f"{'NAME':30s} {'TYPE':10s} {'BASE(S)':10s} {'DESCRIPTION':40s} ORIGINAL SOURCE")
    for model in model_list:
        print(f"{model.name:30s} {model.type.value:10s} {', '.join([x.value for x in model.base_models]):10s} {model.description:40s} {model.source}")


def ingest_models(args):
    for path in args.model_paths:
        try:
            print(f'ingesting {path}...', end='')
            model_manager.ingest(path)
            print('success.')
        except (OSError, InvalidModelException, DuplicateModelException) as e:
            print(f'FAILED: {e}')


def export_model(args):
    print(f'exporting {args.model_name} to {args.destination}...', end='')
    try:
        model_manager.export_pipeline(args.model_name, args.destination)
        print('success.')
    except (OSError, ModelNotFoundException, InvalidModelException) as e:
        print(f'FAILED: {e}')

def main():
    global model_manager
    global config

    parser = argparse.ArgumentParser(description="Normalized model manager util")
    parser.add_argument('--root_dir',
                        dest="root",
                        type=str,
                        default=None,
                        help="path to INVOKEAI_ROOT"
                        )
    subparsers = parser.add_subparsers(help="commands")
    parser_ingest = subparsers.add_parser('ingest', help='ingest checkpoint or diffusers models')
    parser_ingest.add_argument('model_paths',
                               type=Path,
                               nargs='+',
                               help='paths to one or more models to be ingested'
                               )
    parser_ingest.set_defaults(func=ingest_models)

    parser_export = subparsers.add_parser('export', help='export a pipeline  to indicated directory')
    parser_export.add_argument('model_name',
                               type=str,
                               help='name of model to export',
                               )
    parser_export.add_argument('destination',
                               type=Path,
                               help='path to destination to export pipeline to',
                               )
    parser_export.set_defaults(func=export_model)

    parser_list = subparsers.add_parser('list', help='list models')
    parser_list.set_defaults(func=list_models)

    parser_listparts = subparsers.add_parser('list-parts', help='list the parts of a pipeline model')
    parser_listparts.add_argument('model_name',
                               type=str,
                               help='name of pipeline model to list parts of',
                               )
    parser_listparts.set_defaults(func=list_parts)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args = parser.parse_args()
    if args.root:
        config.parse_args(['--root', args.root])
    else:
        config.parse_args([])

    model_manager = NormalizedModelManager(config)
    args.func(args)


if __name__ == '__main__':
    main()
