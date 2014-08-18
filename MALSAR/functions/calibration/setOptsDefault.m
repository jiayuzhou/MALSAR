function [ opts ] = setOptsDefault( opts, field, defaultValue)
%SETOPTS Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts, field)
        opts.(field) = defaultValue;
    end

end

