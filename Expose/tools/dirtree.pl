#!/usr/bin/perl -w

# as from https://texblog.org/2012/08/07/semi-automatic-directory-tree-in-latex/
# USAGES:
#  perl dirtree.pl /path/to/top/of/dirtree
#  perl dirtree.pl /path/to/top/of/dirtree > /path/to/output/file


use strict;
use File::Find;

# get toplevel from argv
my $top = shift @ARGV;
# check if $top is given
die "specify top directory\n" unless defined $top;
# check if $top is a existing directory
chdir $top or die "cannot chdir to $top: $!\n";
# recursive exploring
find(sub {
    local $_ = $File::Find::name;
    my @F = split '/';
    printf ".%d %s.\n", scalar @F, @F==1 ? $top : $F[-1];
}, '.');
