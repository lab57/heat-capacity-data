function multiObjectTracking(filename)
  % Create System objects used for reading video, detecting moving objects,
  % and displaying the results.
  obj = setupSystemObjects();
  tracks = initializeTracks(); % Create an empty array of tracks.
  nextId = 1; % ID of the next track
  % Detect moving objects, and track them across video frames.

   % Initialize a cell array to store centroids; this will grow dynamically
    centroidData = {};
  


  while ~isDone(obj.reader)
      frame = readFrame();
      [centroids, bboxes, mask] = detectObjects(frame);
      predictNewLocationsOfTracks();
      [assignments, unassignedTracks, unassignedDetections] = ...
          detectionToTrackAssignment();
      updateAssignedTracks();

% Collect centroids for all tracks in the current frame
currentFrameCentroids = NaN(size(tracks, 2), 2); % Assuming tracks are stored in a 1xN structure array
for i = 1:length(tracks)
    track = tracks(i);
    % Check if the track was updated in this frame
    if track.totalVisibleCount - track.consecutiveInvisibleCount > 0
        currentFrameCentroids(i, :) = track.kalmanFilter.State([1, 3])'; % Assuming the state vector format [x, dx, y, dy]
    end
end
% Append to the centroidData
centroidData{end+1} = currentFrameCentroids;


      updateUnassignedTracks();
      deleteLostTracks();
      createNewTracks();
      displayTrackingResults();
  end

% Initialize CSV data array with NaNs to handle frames with varying numbers of tracks
maxNumOfTracks = max(cellfun(@(c) size(c, 1), centroidData)); % Maximum number of tracks in any frame
numOfFrames = length(centroidData); % Total number of frames
% Each track has 2 values (x, y), hence maxNumOfTracks * 2
csvData = NaN(numOfFrames, maxNumOfTracks * 2);
releaseSystemObjects(obj);
close all;

for i = 1:numOfFrames
    frameData = centroidData{i};
    if ~isempty(frameData)
        % Flatten frameData from Mx2 to 1x(2*M) and assign to csvData
        reshapedData = reshape(frameData', 1, []);
        csvData(i, 1:numel(reshapedData)) = reshapedData;
    end
end

% Write to CSV, adjusting the filename to your needs
csvwrite(filename + '.csv', csvData);
   function obj = setupSystemObjects()
          % Initialize Video I/O
          % Create objects for reading a video from a file, drawing the tracked
          % objects in each frame, and playing the video.
          % Create a video file reader.s
          obj.reader = vision.VideoFileReader(filename + '.mp4');
          % Create two video players, one to display the video,
          % and one to display the foreground mask.
          obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 400, 1300]);
          obj.maskPlayer = vision.VideoPlayer('Position', [420, 400, 400, 1300]);
          % Create System objects for foreground detection and blob analysis
          % The foreground detector is used to segment moving objects from
          % the background. It outputs a binary mask, where the pixel value
          % of 1 corresponds to the foreground and the value of 0 corresponds
          % to the background.
          obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
              'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.65);
          % Connected groups of foreground pixels are likely to correspond to moving
          % objects.  The blob analysis System object is used to find such groups
          % (called 'blobs' or 'connected components'), and compute their
          % characteristics, such as area, centroid, and the bounding box.
          obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
              'AreaOutputPort', true, 'CentroidOutputPort', true, ...
              'MinimumBlobArea', 50, 'MaximumBlobArea', 2000);
   end
function releaseSystemObjects(obj)
    % Release resources.
    release(obj.reader);
    release(obj.videoPlayer);
    release(obj.maskPlayer);
    release(obj.detector);
    release(obj.blobAnalyser);
end
   function tracks = initializeTracks()
          % create an empty array of tracks
          tracks = struct(...
              'id', {}, ...
              'bbox', {}, ...
              'kalmanFilter', {}, ...
              'age', {}, ...
              'totalVisibleCount', {}, ...
              'consecutiveInvisibleCount', {});
   end
   function frame = readFrame()
          frame = obj.reader.step();
   end
   function [centroids, bboxes, mask] = detectObjects(frame)
          % Detect foreground.
          mask = obj.detector.step(frame);
          % Apply morphological operations to remove noise and fill in holes.
          mask = imopen(mask, strel('rectangle', [3,3]));
          mask = imclose(mask, strel('rectangle', [15, 15]));
          mask = imfill(mask, 'holes');
          % Perform blob analysis to find connected components.
          [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
   end
   function predictNewLocationsOfTracks()
          for i = 1:length(tracks)
              bbox = tracks(i).bbox;
              % Predict the current location of the track.
              predictedCentroid = predict(tracks(i).kalmanFilter);
              % Shift the bounding box so that its center is at
              % the predicted location.
              predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
              tracks(i).bbox = [predictedCentroid, bbox(3:4)];
          end
   end
   function [assignments, unassignedTracks, unassignedDetections] = ...
              detectionToTrackAssignment()
          nTracks = length(tracks);
          nDetections = size(centroids, 1);
          % Compute the cost of assigning each detection to each track.
          cost = zeros(nTracks, nDetections);
          for i = 1:nTracks
              cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
          end
          % Solve the assignment problem.
          costOfNonAssignment = 20;
          [assignments, unassignedTracks, unassignedDetections] = ...
              assignDetectionsToTracks(cost, costOfNonAssignment);
   end
  function updateAssignedTracks()
          numAssignedTracks = size(assignments, 1);
          for i = 1:numAssignedTracks
              trackIdx = assignments(i, 1);
              detectionIdx = assignments(i, 2);
              centroid = centroids(detectionIdx, :);
              bbox = bboxes(detectionIdx, :);
              % Correct the estimate of the object's location
              % using the new detection.
              correct(tracks(trackIdx).kalmanFilter, centroid);
              % Replace predicted bounding box with detected
              % bounding box.
              tracks(trackIdx).bbox = bbox;
              % Update track's age.
              tracks(trackIdx).age = tracks(trackIdx).age + 1;
              % Update visibility.
              tracks(trackIdx).totalVisibleCount = ...
                  tracks(trackIdx).totalVisibleCount + 1;
              tracks(trackIdx).consecutiveInvisibleCount = 0;
          end
  end
   function updateUnassignedTracks()
          for i = 1:length(unassignedTracks)
              ind = unassignedTracks(i);
              tracks(ind).age = tracks(ind).age + 1;
              tracks(ind).consecutiveInvisibleCount = ...
                  tracks(ind).consecutiveInvisibleCount + 1;
          end
   end
  function deleteLostTracks()
          if isempty(tracks)
              return;
          end
          invisibleForTooLong = 20;
          ageThreshold = 8;
          % Compute the fraction of the track's age for which it was visible.
          ages = [tracks(:).age];
          totalVisibleCounts = [tracks(:).totalVisibleCount];
          visibility = totalVisibleCounts ./ ages;
          % Find the indices of 'lost' tracks.
          lostInds = (ages < ageThreshold & visibility < 0.6) | ...
              [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
          % Delete lost tracks.
          tracks = tracks(~lostInds);
  end
   function createNewTracks()
          centroids = centroids(unassignedDetections, :);
          bboxes = bboxes(unassignedDetections, :);
          for i = 1:size(centroids, 1)
              centroid = centroids(i,:);
              bbox = bboxes(i, :);
              % Create a Kalman filter object.
              kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                  centroid, [200, 50], [100, 25], 100);
              % Create a new track.
              newTrack = struct(...
                  'id', nextId, ...
                  'bbox', bbox, ...
                  'kalmanFilter', kalmanFilter, ...
                  'age', 1, ...
                  'totalVisibleCount', 1, ...
                  'consecutiveInvisibleCount', 0);
              % Add it to the array of tracks.
              tracks(end + 1) = newTrack;
              % Increment the next id.
              nextId = nextId + 1;
          end
   end
   function displayTrackingResults()
          % Convert the frame and the mask to uint8 RGB.
          frame = im2uint8(frame);
          mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
          minVisibleCount = 8;
          if ~isempty(tracks)
              % Noisy detections tend to result in short-lived tracks.
              % Only display tracks that have been visible for more than
              % a minimum number of frames.
              reliableTrackInds = ...
                  [tracks(:).totalVisibleCount] > minVisibleCount;
              reliableTracks = tracks(reliableTrackInds);
              % Display the objects. If an object has not been detected
              % in this frame, display its predicted bounding box.
              if ~isempty(reliableTracks)
                  % Get bounding boxes.
                  bboxes = cat(1, reliableTracks.bbox);
                  % Get ids.
                  ids = int32([reliableTracks(:).id]);
                  % Create labels for objects indicating the ones for
                  % which we display the predicted rather than the actual
                  % location.
                  labels = cellstr(int2str(ids'));
                  predictedTrackInds = ...
                      [reliableTracks(:).consecutiveInvisibleCount] > 0;
                  isPredicted = cell(size(labels));
                  isPredicted(predictedTrackInds) = {' predicted'};
                  labels = strcat(labels, isPredicted);
                  % Draw the objects on the frame.
                  frame = insertObjectAnnotation(frame, 'rectangle', ...
                      bboxes, labels);
                  % Draw the objects on the mask.
                  mask = insertObjectAnnotation(mask, 'rectangle', ...
                      bboxes, labels);
              end
          end
          % Display the mask and the frame.
          obj.maskPlayer.step(mask);
          obj.videoPlayer.step(frame);
   end
  end