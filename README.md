# spike-the-biker
winning code for comma hack 4 - an indoor navigation algorithms to make a human-size robot move around the office!

## how it works: sim2real + RL + classifier
- We scan the whole office using LiDAR. We then drew a 2D vector floor plan, in which we created a path to follow.
- We then tuned a PID to follow the path whenever an agent spawns on the office. We sample and conjoin the trajectories, then use this desired trajectory to get back to the line as the ground truth for the model.

![](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/608/740/datas/original.gif)

- Then we used the trajectories of the PID to create keyframe for a camera in a blender project of the scanned office. This way we have a pair of generated images / actions, that we use to train an EfficientNet that will run on-device. The EfficientNet uses the raw image to predict the action and trajectory for the next timesteps.

![](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/608/704/datas/original.gif)

(note that this setup/approach works given any 3d render of an environment and a desired trajectory)

## demo

https://github.com/expofx/spike-the-biker/assets/56745453/2734792b-728b-41f2-beb6-cc5113c3907a
