%% FUNCTION init_opts
% initialization options for multi-task learning library
%
% If one of the ncessary opts are empty then it will be set to default
% values as specified in this file.
%
% Table of Options.  * * indicates default value.
% FIELD                DESCRIPTION
%% Optimization options
%
%  .max_iter               Maximum iteration step number
%                           *1000*
%  .tol                    Tolerance
%                           *10e-3*
%  .tFlag                  Termination condition
%                           0 => change of absolute function value:
%                             abs( funcVal(i)- funcVal(i-1) ) <= .tol
%                         * 1 => change of relative function value:
%                             abs( funcVal(i)- funcVal(i-1) )
%                              <= .tol * funcVal(i-1)
%                           2 => absolute function value:
%                             funcVal(end)<= .tol
%                           3 => Run the code for .maxIter iterations
%% Starting Point
%
% .W0               Starting point of W.
%                   Initialized according to .init.
%
% .C0               Starting point for the intercept C (for Logistic Loss)
%                   Initialized according to .init.
%
% .init             .init specifies how to initialize W, C.
%                         0 => .W0, C0 are set by a guess value infered from data
%                         1 => .W0 and .C0 are defined
%                       * 2 => .W0= zeros(.), .C0=0 *
%
%% Parallel Computing
%
% .pFlag            Enable Map-Reduce (needs Parallel Toolbox support).
%                        *false*
%
% .pSeg_num         set the number of total parallel segmentations. if the
%                   number is non-positive, then current matlab pool number
%                   will be used. [supported in the incoming version]
%                        * pool size *
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
%   Last modified on June 21, 2012.
%

function opts = init_opts (opts)

%% Default values
DEFAULT_MAX_ITERATION = 1000;
DEFAULT_TOLERANCE     = 1e-4;
MINIMUM_TOLERANCE     = eps * 100;
DEFAULT_TERMINATION_COND = 1;
DEFAULT_INIT = 2;
DEFAULT_PARALLEL_SWITCH = false;

%% Starting Point
if isfield(opts,'init')
    if (opts.init~=0) && (opts.init~=1) && (opts.init~=2)
        opts.init=DEFAULT_INIT; % if .init is not 0, 1, or 2, then use the default 0
    end
    
    if (~isfield(opts,'W0')) && (opts.init==1)
        opts.init=DEFAULT_INIT; % if .W0 is not defined and .init=1, set .init=0
    end
else
    opts.init = DEFAULT_INIT; % if .init is not specified, use "0"
end

%% Tolerance
if isfield(opts, 'tol')
    % detect if the tolerance is smaller than minimum
    % tolerance allowed.
    if (opts.tol <MINIMUM_TOLERANCE)
        opts.tol = MINIMUM_TOLERANCE;
    end
else
    opts.tol = DEFAULT_TOLERANCE;
end

%% Maximum iteration steps
if isfield(opts, 'maxIter')
    if (opts.maxIter<1)
        opts.maxIter = DEFAULT_MAX_ITERATION;
    end
else
    opts.maxIter = DEFAULT_MAX_ITERATION;
end

%% Termination condition
if isfield(opts,'tFlag')
    if opts.tFlag<0
        opts.tFlag=0;
    elseif opts.tFlag>3
        opts.tFlag=3;
    else
        opts.tFlag=floor(opts.tFlag);
    end
else
    opts.tFlag = DEFAULT_TERMINATION_COND;
end

%% Parallel Options
if isfield(opts, 'pFlag')
    if opts.pFlag == true && ~exist('matlabpool', 'file')
        opts.pFlag = false;
        warning('MALSAR:PARALLEL','Parallel Toolbox is not detected, MALSAR is forced to turn off pFlag.');
    elseif opts.pFlag ~= true && opts.pFlag ~= false
        % validate the pFlag.
        opts.pFlag = DEFAULT_PARALLEL_SWITCH;
    end
else
    % if not set the pFlag to default.
    opts.pFlag = DEFAULT_PARALLEL_SWITCH;
end

if opts.pFlag
    % if the pFlag is checked,
    % check segmentation number.
    if isfield(opts, 'pSeg_num')
        if opts.pSeg_num < 0
            opts.pSeg_num = matlabpool('size');
        else
            opts.pSeg_num = ceil(opts.pSeg_num);
        end
    else
        opts.pSeg_num = matlabpool('size');
    end
end
