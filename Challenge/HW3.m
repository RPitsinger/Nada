%% Q1

% Clear previous data and figures
clc;
clear;
close all;

% Common Parameters
x = linspace(-pi, pi, 400); % Points to evaluate the series

% First Series: Complex Exponential with Fourier Coefficients
N1 = 25; % Number of terms on each side of zero, excluding zero
sum_series1 = zeros(size(x));
for n = -N1:N1
    if n ~= 0
        term1 = (1 - (-1)^n) / (i * n * pi);
        sum_series1 = sum_series1 + term1 * exp(i * n * x);
    end
end

% Second Series: Sine Series %
sum_series2 = zeros(size(x));
for n = 1:N1
    sum_series2 = sum_series2 + (2 / (n * pi)) * (1 - (-1)^n) * sin(n * x);
end

% Plotting both series on the same graph
figure;
hold on;
plot(x, real(sum_series1), 'LineWidth', 3, 'DisplayName', 'Complex Fourier Series');
plot(x, sum_series2, '--', 'LineWidth', 3, 'DisplayName', 'Normal Fourier Series');
hold off;
title('Comparison of Fourier Series');
xlabel('x');
ylabel('Sum of Series');
legend show;
grid on;

%% 4A Cos
% Define the original function and Fourier cosine transform in compact form
f_original = @(x) exp(-2*x) .* cos(x);
A_cosine = @(omega) (4/pi) * (5 + omega.^2) ./ (25 + 6*omega.^2 + omega.^4);
A_integral = @(omega) (2/pi) * ((omega.^2 + 5) ./ (omega.^4 + 10*omega.^2 + 21));

% Fourier cosine integral approximation
f_fourier = @(x) integral(@(omega) A_cosine(omega) .* cos(omega * x), 0, Inf);

% Compute the original function and Fourier approximation for a specific x
x_val = 1;
f_val_original = f_original(x_val);
f_val_fourier = f_fourier(x_val);

% Compute the second integral with a finite upper bound for omega
omega_max = 100;
result_integral = integral(@(omega) A_integral(omega) .* cos(omega * x_val), 0, omega_max);

% Display the results
fprintf('Original function value at x = %f: %f\n', x_val, f_val_original);
fprintf('Fourier cosine integral approximation at x = %f: %f\n', x_val, f_val_fourier);
fprintf('Solution integral for x = %f is %f\n', x_val, result_integral);

%% 
% Set x value and omega upper limit
x_val = 1;          % You can vary this value
omega_max = 100000; % Upper limit for omega

% Define the sine transform A_s(omega) and A(omega)
A_s = @(omega) (2*omega/pi) .* ((3 + omega.^2) ./ (25 + 6*omega.^2 + omega.^4));
A = @(omega) (2/pi) * ((omega.^3 + 1) ./ (omega.^4 + 10*omega.^2 + 21));

% Compute the sine integrals numerically
result_A_s = integral(@(omega) A_s(omega) .* sin(omega * x_val), 0, omega_max);
result_A = integral(@(omega) A(omega) .* sin(omega * x_val), 0, omega_max);

% Define the original function f(x) = e^(-2x) * cos(x)
f_original = @(x) exp(-2*x) .* cos(x);
f_val_original = f_original(x_val);  % Compute the original function at x_val

% Display the results
fprintf('Sine integral (A_s) approximation at x = %f: %f\n', x_val, result_A_s);
fprintf('Sine integral (Solution) approximation at x = %f: %f\n', x_val, result_A);
fprintf('Original function value at x = %f: %f\n', x_val, f_val_original);

%% 
% Define the frequency range
omega = linspace(-20, 20, 400);

% Define the amplitude spectrum function
amplitude_spectrum = 24 ./ (16 + omega.^2);

% Plot the amplitude spectrum
figure;
plot(omega, amplitude_spectrum, 'LineWidth', 2);
title('Amplitude Spectrum');
xlabel('\omega');
ylabel('|f(\omega)|');
grid on;

%% 
% Define constants
k = 2; % You can change this value depending on the problem context

% Define frequency range
omega = linspace(-10, 10, 1000); % Frequency values from -10 to 10

% Compute the expression for the imaginary part
numerator1 = sin(k * (omega + 1));
numerator2 = sin(k * (omega - 1));
denominator1 = omega + 1;
denominator2 = omega - 1;

% Avoid division by zero
denominator1(denominator1 == 0) = eps; % Replace zero values with a small value eps
denominator2(denominator2 == 0) = eps; % Same here

% Compute the imaginary part
imaginary_part = (numerator1 ./ denominator1) - (numerator2 ./ denominator2);

% Compute the amplitude spectrum (absolute value of imaginary part)
amplitude_spectrum = abs(imaginary_part);

% Plot the amplitude spectrum
figure;
plot(omega, amplitude_spectrum, 'LineWidth', 2);
xlabel('\omega');
ylabel('|F(\omega)|');
title('Amplitude Spectrum');
grid on;

%% 
%% 
% Set x value and omega upper limit
x_val = 0.5;          % You can vary this value
omega_max = 100000; % Upper limit for omega

% Define the first function A_s(omega) based on the provided integral form
A_s = @(omega) (16/pi) * (omega ./ (omega.^2 + 16).^2);

% Compute the sine transform numerically for A_s
result_A_s = integral(@(omega) A_s(omega) .* sin(omega * x_val), 0, omega_max);

% Define the original function f(x) = x * e^(-4|x|)
f_original = @(x) x .* exp(-4 * abs(x));
f_val_original = f_original(x_val);  % Compute the original function at x_val

% Display the results
fprintf('Sine integral (A_s) approximation at x = %f: %f\n', x_val, result_A_s);
fprintf('Original function value at x = %f: %f\n', x_val, f_val_original);

% Check if they are approximately equal
if abs(result_A_s - f_val_original) < 1e-6
    fprintf('The two expressions are approximately equal at x = %f.\n', x_val);
else
    fprintf('The two expressions are NOT equal at x = %f.\n', x_val);
end