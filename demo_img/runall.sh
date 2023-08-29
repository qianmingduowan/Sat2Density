for case in `ls -d demo_img/case*`
do
    echo $case
    python test.py --yaml=sat2density_cvact \
    --test_ckpt_path=2u87bj8w \
    --task=test_vid \
    --demo_img=$case/satview-input.png  \
    --sty_img=$case/groundview.image.png  \
    --save_dir=results/$case
    ffmpeg -framerate 10 -i results/$case/rendered_images+depths/%5d.png results/$case/render.gif
    ffmpeg -framerate 10 -i results/$case/sat_images/%5d.png results/$case/sat.gif
done

for case in `ls -d demo_img/case*`
do
    echo $case
    sat_gif=results/$case/sat.gif
    render_gif=results/$case/render.gif
    cp $sat_gif docs/figures/$case.sat.gif
    cp $render_gif docs/figures/$case.render.gif
done