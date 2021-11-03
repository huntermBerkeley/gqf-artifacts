# GQF: A Practical Counting Quotient Filter for GPUs


Overview
--------
 The GQF supports approximate membership testing and counting the occurrences of
 items in a data set. This general-purpose AMQ is small and fast, has good
 locality of reference, and supports deletions,
 counting (even on skewed data sets), resizing, and highly concurrent
 access.

API
--------

* \_\_host\_\_ void qf_malloc_device(QF** qf, int nbits): Initializes a new GQF with 2^nbits slots, qf is set to point to the new filter
* \_\_host\_\_ void qf_destroy_device(QF * qf): Frees the GQF pointed to by qf.

POINT API
--------


* \_\_device\_\_ qf_returns point_insert(QF* qf, uint64_t key, uint8_t value, uint8_t flags): Insert an ittem into the filter.
* \_\_device\_\_ qf_returns point_insert_not_exists(QF* qf, uint64_t key, uint8_t value, uint8_t& returnedVal,  uint8_t flags): Check if an item is found in the filter. If so, place the item in returnedVal and return QF_ITEM_FOUND. If the item has not been seen before, insert it into the filter and return QF_ITEM_INSERTED. If the QF is full, return QF_FULL.
* \_\_device\_\_ uint64_t point_query(QF* qf, uint64_t key, uint8_t value, uint8_t& returnedVal, uint8_t flags): Return the count of an item in the filter, return 0 if the itemis not found.
* \_\_device\_\_ uint64_t point_query_concurrent(QF* qf, uint64_t key, uint8_t value, uint8_t& returnedVal, uint8_t flags): Same behavior as point_query, but with locking. Use this when inserts and queries must occur simultaneously and counts are required. (If counts are not necessary, point_insert_not_exists is faster)



BULK API
--------
* \_\_host\_\_ void bulk_insert(QF* qf, uint64_t* items, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes): Insert a batch of items into the filter using the even-odd insert scheme.
* \_\_host\_\_ void bulk_insert_reduce(QF* qf, uint64_t* keys, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes): Insert a batch of items, but perform a reduction before inserting into the CQF. This should be used when the inputs are expected to have heavy skew.
* \_\_host\_\_ void bulk_get(QF* qf, uint64_t * keys, uint64_t nvals, uint64_t * returns): Fills returns with the counts of keys in the filter.






Build
-------
This library depends on [Thrust](https://thrust.github.io/). 

In addition, one of the filters available for testing, the SQF, depends on [CUB](https://nvlabs.github.io/cub/) and [ModernGPU](https://moderngpu.github.io/intro.html). 

The code uses two new instructions to implement select on machine words introduced 
in intel's Haswell line of CPUs. However, there is also an alternate implementation
of select on machine words to work on CPUs older than Haswell.

To build:
```bash
 $ source modules.sh
 $ make test
 $ ./test -n 28 -d filter_type
```


The argument to -n is the log of the number of slots in the GQF. For example,
 to create a CQF with 2^30 slots, the argument will be -n 30.

The argument to -d is the filter being tested. The currently supported filter_typescd  are:

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
