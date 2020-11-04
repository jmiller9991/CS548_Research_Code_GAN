# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import projector
import pretrained_networks
from training import dataset
from training import misc

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots):
    print('Starting projecting of image.')
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    print('Snapshotted steps')
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    print('saved image as grid')
    proj.start(targets)
    print('Started projector')
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        print('Step done')
        if proj.get_cur_step() in snapshot_steps:
            print('Cond Reached')
            sub_dalatents = proj.get_dlatents()
            np.save(png_prefix + 'step%04d.png' % proj.get_cur_step(), sub_dalatents)
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
            print('Saved Step Image')
        dalatents = proj.get_dlatents()
        print(type(dalatents))
        print(dalatents.shape)
        np.save(png_prefix + "final_dlatent", dalatents)

    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, truncation_psi):
    print('GENERATED: Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    print('Networks Loaded')
    proj = projector.Projector()
    print('Projector Obtained')
    proj.set_network(Gs)
    print('Projector Network Set')
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    print('Obtained noise')

    print('Getting Generator S arguments')
    Gs_kwargs = dnnlib.EasyDict()
    print('Gs_kwargs received')
    Gs_kwargs.randomize_noise = False
    print('Gs_kwargs noise randomization is ' + str(Gs_kwargs.randomize_noise))
    Gs_kwargs.truncation_psi = truncation_psi
    print('Gs_kwargs.truncation_pis set')

    for seed_idx, seed in enumerate(seeds):
        print('GENERATED: Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        print('Producing random values')
        z = rnd.randn(1, *Gs.input_shape[1:])
        print('Producing z')
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        print('Setting vars in tflib')
        images = Gs.run(z, None, **Gs_kwargs)
        print('Running Gs on images')
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)
        print('Projected image from ' + str(seed_idx))

#----------------------------------------------------------------------------

def project_real_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots):
    print('REAL: Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    print('Networks Loaded')
    proj = projector.Projector()
    print('Projector Obtained')
    proj.set_network(Gs)
    print('Projector Network Set')

    print('REAL: Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    print('Loaded Dataset')
    assert dataset_obj.shape == Gs.output_shape[1:]
    print('dataset_obj_shape set to Gs output shape')

    for image_idx in range(num_images):
        print('REAL: Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        print('Obtained image and label from dataset' + str(image_idx))
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        print('Adjusted Dynamic Ranges')
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)
        print('image ' + str(image_idx) + ' projected')

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals if x]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 2
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_projector.project_generated_images',
        'project-real-images': 'run_projector.project_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
