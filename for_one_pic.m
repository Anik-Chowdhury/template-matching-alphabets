clc;
clear;
close all;

% 1. Preprocessing
candidate_filename = 'Dd.png'; % Candidate image filename
candidate_image = imread(candidate_filename);
template_image = imread('alphabet_grid.png'); % Load template image

candidate_gray = rgb2gray(candidate_image);
template_gray = rgb2gray(template_image);

threshold_candidate = graythresh(candidate_gray); % Otsu's threshold for candidate
threshold_template = graythresh(template_gray); % Otsu's threshold for template

candidate_binary = imbinarize(candidate_gray, threshold_candidate); % Otsu's thresholding
template_binary = imbinarize(template_gray, threshold_template);

% Invert colors (black characters on white background)
candidate_binary = ~candidate_binary;
template_binary = ~template_binary;

% 2. Template Segmentation
[template_rows, template_cols] = size(template_binary);
grid_rows = 5; % 5 rows in the template grid
grid_cols = 6; % 6 columns in the template grid

segment_rows = floor(template_rows / grid_rows);
segment_cols = floor(template_cols / grid_cols);

candidate_resized = imresize(candidate_binary, [segment_rows, segment_cols]); % Resize candidate

% 3. Normalized Cross-Correlation (NCC) Calculation
ncc_values = zeros(grid_rows, grid_cols);
best_ncc = -1;
best_row = 0;
best_col = 0;

for row = 1:grid_rows
    for col = 1:grid_cols
        row_start = (row - 1) * segment_rows + 1;
        row_end = row * segment_rows;
        col_start = (col - 1) * segment_cols + 1;
        col_end = col * segment_cols;

        template_segment = template_binary(row_start:row_end, col_start:col_end);
        correlation_matrix = normxcorr2(candidate_resized, template_segment); % Store the matrix.
        ncc_values(row, col) = max(correlation_matrix(:)); % Take the max value.

        if ncc_values(row, col) > best_ncc
            best_ncc = ncc_values(row, col);
            best_row = row;
            best_col = col;
        end
    end
end

% 4. Load Ground Truth and Performance Evaluation
ground_truth = readtable('ground_truth.txt', 'Delimiter', ',');
ground_truth_row = ground_truth.Row(strcmp(ground_truth.Filename, candidate_filename));
ground_truth_col = ground_truth.Column(strcmp(ground_truth.Filename, candidate_filename));

% Check if ground truth exists
if isempty(ground_truth_row) || isempty(ground_truth_col)
    warning(['No ground truth found for ', candidate_filename]);
    ground_truth_row = -1; % Assign an invalid value
    ground_truth_col = -1;
end

% Initialize performance metrics
TP = 0;
FP = 0;
FN = 0;
TN = 0;

if best_row == ground_truth_row && best_col == ground_truth_col
    TP = 1;
else
    FP = 1;
    FN = 1; % Since there is only one correct match
end

% Compute TN
TN = (grid_rows * grid_cols) - TP - FP - FN;

% Compute Performance Metrics (Preventing Division by Zero)
if (TP + FP) == 0
    Precision = 0;
else
    Precision = TP / (TP + FP);
end

if (TP + FN) == 0
    Recall = 0;
else
    Recall = TP / (TP + FN);
end

Accuracy = (TP + TN) / (TP + TN + FP + FN);

% 5. Visualization
figure;
subplot(2, 2, 1);
imshow(template_binary); % Invert template display
title('Template Image (Inverted)');

subplot(2, 2, 2);
imshow(candidate_binary); % Invert candidate display
title('Candidate Image (Inverted)');

subplot(2, 2, 3);
imshow(template_binary); % Invert template display
hold on;
rectangle('Position', [(best_col - 1) * segment_cols + 1, (best_row - 1) * segment_rows + 1, segment_cols, segment_rows], 'EdgeColor', 'r', 'LineWidth', 2);
title('Best Match Highlighted (Inverted)');
hold off;

subplot(2, 2, 4);
bar([Precision, Recall, Accuracy]);
title('Performance Metrics');
set(gca, 'xticklabel', {'Precision', 'Recall', 'Accuracy'});

disp(['Best Match: Row ', num2str(best_row), ', Column ', num2str(best_col)]);
disp(['Precision: ', num2str(Precision)]);
disp(['Recall: ', num2str(Recall)]);
disp(['Accuracy: ', num2str(Accuracy)]);
disp(['Best NCC Value: ', num2str(best_ncc)]);
disp(['Candidate Threshold Value: ', num2str(threshold_candidate)]);
disp(['Template Threshold Value: ', num2str(threshold_template)]);

% Display all NCC values as a table
disp('NCC Values Table:');
for row = 1:grid_rows
    for col = 1:grid_cols
        fprintf('%-10.5f', ncc_values(row, col)); % Format each value
    end
    fprintf('\n'); % New line after each row
end
