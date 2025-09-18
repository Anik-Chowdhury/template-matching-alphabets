clc;
clear;
close all;
% Character Recognition for Multiple Candidate Images with Overall Metrics

% 1. Setup
template_image = imread('alphabet_grid.png');
template_gray = rgb2gray(template_image);
template_binary = imbinarize(template_gray, graythresh(template_gray));
template_binary = ~template_binary; % Invert image

[template_rows, template_cols] = size(template_binary);
grid_rows = 5;
grid_cols = 6;
segment_rows = floor(template_rows / grid_rows);
segment_cols = floor(template_cols / grid_cols);

candidate_folder = 'candidates';
candidate_files = dir(fullfile(candidate_folder, '*.png'));

% Initialize performance metrics
total_TP = 0;
total_FP = 0;
total_FN = 0;

% Read ground truth table
ground_truth = readtable('ground_truth.txt', 'Delimiter', ',');

% 2. Loop through candidate images
for file_idx = 1:length(candidate_files)
    candidate_filename = candidate_files(file_idx).name;
    candidate_image = imread(fullfile(candidate_folder, candidate_filename));

    candidate_gray = rgb2gray(candidate_image);
    candidate_binary = imbinarize(candidate_gray, graythresh(candidate_gray));
    candidate_binary = ~candidate_binary; % Invert image

    candidate_resized = imresize(candidate_binary, [segment_rows, segment_cols]);

    % 3. NCC Calculation
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
            correlation_matrix = normxcorr2(candidate_resized, template_segment);
            ncc_values(row, col) = max(correlation_matrix(:));

            if ncc_values(row, col) > best_ncc
                best_ncc = ncc_values(row, col);
                best_row = row;
                best_col = col;
            end
        end
    end

    % 4. Ground Truth and Performance Calculation
    ground_truth_row = ground_truth.Row(strcmp(ground_truth.Filename, candidate_filename));
    ground_truth_col = ground_truth.Column(strcmp(ground_truth.Filename, candidate_filename));

    TP = 0;
    FP = 0;
    FN = 0;

    if isempty(ground_truth_row) || isempty(ground_truth_col)
        warning(['No ground truth found for ', candidate_filename]);
        continue;
    end

    % Correct detection
    if best_row == ground_truth_row && best_col == ground_truth_col
        TP = 1;
    else
        FP = 1;
        FN = 1; % Incorrect guess means we missed the actual match
    end

    % Update global counts
    total_TP = total_TP + TP;
    total_FP = total_FP + FP;
    total_FN = total_FN + FN;

    % 5. Per-Image Precision, Recall, Accuracy (Fixing division by zero)
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

    Accuracy = (TP + ((grid_rows * grid_cols) - (TP + FP + FN))) / (grid_rows * grid_cols);
    % 6. Visualization
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


    % 7. Display Results
    disp(['Best Match: Row ', num2str(best_row), ', Column ', num2str(best_col)]);
    disp(['Precision: ', num2str(Precision)]);
    disp(['Recall: ', num2str(Recall)]);
    disp(['Accuracy: ', num2str(Accuracy)]);
    disp(['Best NCC Value: ', num2str(best_ncc)]);

    % Display NCC values as a table
    disp('NCC Values Table:');
    for row = 1:grid_rows
        for col = 1:grid_cols
            fprintf('%-10.5f', ncc_values(row, col)); % Format each value
        end
        fprintf('\n'); % New line after each row
    end
end

% 8. Calculate Overall Metrics
if (total_TP + total_FP) == 0
    Precision = 0;
else
    Precision = total_TP / (total_TP + total_FP);
end

if (total_TP + total_FN) == 0
    Recall = 0;
else
    Recall = total_TP / (total_TP + total_FN);
end

Accuracy = total_TP / (total_TP + total_FP + total_FN);

disp(['\nOverall Precision: ', num2str(Precision)]);
disp(['Overall Recall: ', num2str(Recall)]);
disp(['Overall Accuracy: ', num2str(Accuracy)]);
