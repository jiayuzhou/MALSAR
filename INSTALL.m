%% FILE INSTALL.m
%   mex the C files used in the MALSAR package.
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.

clear, clc;
current_path=cd;

%% Output information
%%
fprintf('\n ----------------------------------------------------------------------------');
fprintf('\n The program is mexing the C files. Please wait...');
fprintf('\n If you have problem with mex, you can refer to the help of Matlab.');
fprintf('\n If you cannot solve the problem, please contact with Jiayu Zhou (jiayu.zhou@asu.edu)\n\n');

%% currently, this package uses the following C files 
%%            (in the folder /MALSAR/c_files)


% files in the folder prf_lbm
cd([current_path '/MALSAR/c_files/prf_lbm']);
mex prf_lbm.cpp;

% file in the folder flsa
cd([current_path '/MALSAR/c_files/flsa']);
mex flsa.c;

% file in the folder eplb
cd([current_path '/MALSAR/c_files/eplb']);
mex eplb.c;

cd([current_path '/MALSAR/c_files/calibration']);
mex segADMM_Zstep.c;
mex segL2.c;
mex segL2Proj.c;
mex segSubg_loss.c;

cd([current_path '/MALSAR/c_files/largescale_ops']);
mex -O -largeArrayDims sparse_inp.c
mex -O -largeArrayDims sparse_update.c

%% Output information
%% 
fprintf('\n\n The C files in the folder c_files have been successfully mexed.');
fprintf('\n\n You can now use the functions in the folder MALSAR.');
fprintf('\n You are suggested to read the manual for better using the codes.');
fprintf('\n You are also suggested to run the examples in the folder Examples for these functions.');
fprintf('\n\n These codes are being developed by Jiayu Zhou and Jieping Ye at Arizona State University.');
fprintf('\n If there is any problem, please contact with Jiayu Zhou (jiayu.zhou@asu.edu).');
fprintf('\n\n Thanks!');
fprintf('\n ----------------------------------------------------------------------------\n');

cd(current_path);
