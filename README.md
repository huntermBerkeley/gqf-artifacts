# gqf

GQF: A Practical Counting Quotient Filter for GPUs


Overview
--------
 The CQF supports approximate membership testing and counting the occurrences of
 items in a data set. This general-purpose AMQ is small and fast, has good
 locality of reference, and supports deletions,
 counting (even on skewed data sets), resizing, and highly concurrent
 access.

API
--------

* \_\_host\_\_ void qf_malloc_device(QF** qf, int nbits): Initializes a new GQF with 2^nbits slots, qf is set to point to the new filter
* \_\_host\_\_ void qf_destroy_device(QF * qf): Frees the GQF pointed to by qf.
* 'qf_insert(item, count)': insert an item to the filter
* 'qf_count_key_value(item)': return the count of the item. Note that this
  method may return false positive results like Bloom filters or an over count.
* 'qf_remove(item, count)': decrement the count of the item by count. If count
  is 0 then completely remove the item.

Build
-------
This library depends on [Thrust](https://thrust.github.io/). 

In addition, one of the filters available for testing, the SQF, depends on [CUB](https://nvlabs.github.io/cub/) and [ModernGPU](https://moderngpu.github.io/intro.html). 

The code uses two new instructions to implement select on machine words introduced 
in intel's Haswell line of CPUs. However, there is also an alternate implementation
of select on machine words to work on CPUs older than Haswell.

To build on a Haswell or newer hardware:
```bash
 $ make test
 $ ./test -n 28 -d filter_type
```


The argument to -n is the log of the number of slots in the GQF. For example,
 to create a CQF with 2^30 slots, the argument will be -n 30.

The argument to -d is the filter being tested. The currently supported filters are:

 - gqf (GQF bulk API)
 - point (GQF point API)
 - sqf (Standard Quotient Filter from [Geil et al.](https://escholarship.org/uc/item/3v12f7dn))
 - rsqf (Rank-Select Quotient Filter from [Geil et al.](https://escholarship.org/uc/item/3v12f7dn))
 - bloom (Bloom Filter)

Contributing
------------
Contributions via GitHub pull requests are welcome.


Authors
-------
- Hunter McCoy <hjmccoy@lbl.gov>
- Prashant Pandey <prashantpa@vmware.com>
