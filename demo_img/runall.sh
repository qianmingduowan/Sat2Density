# for case in `ls -d demo_img/case*`
for case_id in 1 2 3 4
do
    case=demo_img/case$case_id
    echo $case
    python test.py --yaml=sat2density_cvact \
    --test_ckpt_path=2u87bj8w \
    --task=test_vid \
    --demo_img=$case/satview-input.png  \
    --sty_img=$case/groundview.image.png  \
    --save_dir=results/$case
    # ffmpeg -framerate 10 -i results/$case/rendered_images+depths/%5d.png results/$case/render.gif
    ffmpeg -framerate 10 -i results/$case/rendered_images+depths/%5d.png -vf "palettegen" results/$case-palette.png
    ffmpeg -framerate 10 -i results/$case/rendered_images+depths/%5d.png -i results/$case-palette.png -filter_complex "paletteuse" results/$case/render.gif

    ffmpeg -framerate 10 -i results/$case/sat_images/%5d.png -vf "palettegen" results/$case-palette.png
    ffmpeg -framerate 10 -i results/$case/sat_images/%5d.png -i results/$case-palette.png -filter_complex "paletteuse" results/$case/sat.gif
    # ffmpeg -framerate 10 -i results/$case/sat_images/%5d.png results/$case/sat.gif
done

# for case in `ls -d demo_img/case*`
for case_id in 1 2 3 4
do
    case=demo_img/case$case_id
    sat_gif=results/$case/sat.gif
    render_gif=results/$case/render.gif
    # echo $sat_gif
    cp $sat_gif docs/figures/demo/case$case_id.sat.gif
    cp $render_gif docs/figures/demo/case$case_id.render.gif
done