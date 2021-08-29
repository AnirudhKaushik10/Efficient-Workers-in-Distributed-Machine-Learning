/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "pml_ucx_datatype.h"
#include "pml_ucx_request.h"

#include "ompi/runtime/mpiruntime.h"
#include "ompi/attribute/attribute.h"

#include <inttypes.h>
#include <math.h>

#ifdef HAVE_UCP_REQUEST_PARAM_T
#define PML_UCX_DATATYPE_SET_VALUE(_datatype, _val) \
    (_datatype)->op_param.send._val; \
    (_datatype)->op_param.bsend._val; \
    (_datatype)->op_param.recv._val;
#endif

static void* pml_ucx_generic_datatype_start_pack(void *context, const void *buffer,
                                                 size_t count)
{
    ompi_datatype_t *datatype = context;
    mca_pml_ucx_convertor_t *convertor;

    convertor = (mca_pml_ucx_convertor_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.convs);

    OMPI_DATATYPE_RETAIN(datatype);
    convertor->datatype = datatype;
    opal_convertor_copy_and_prepare_for_send(ompi_proc_local_proc->super.proc_convertor,
                                             &datatype->super, count, buffer, 0,
                                             &convertor->opal_conv);
    return convertor;
}

static void* pml_ucx_generic_datatype_start_unpack(void *context, void *buffer,
                                                   size_t count)
{
    ompi_datatype_t *datatype = context;
    mca_pml_ucx_convertor_t *convertor;

    convertor = (mca_pml_ucx_convertor_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.convs);

    OMPI_DATATYPE_RETAIN(datatype);
    convertor->datatype = datatype;
    convertor->offset = 0;
    opal_convertor_copy_and_prepare_for_recv(ompi_proc_local_proc->super.proc_convertor,
                                             &datatype->super, count, buffer, 0,
                                             &convertor->opal_conv);
    return convertor;
}

static size_t pml_ucx_generic_datatype_packed_size(void *state)
{
    mca_pml_ucx_convertor_t *convertor = state;
    size_t size;

    opal_convertor_get_packed_size(&convertor->opal_conv, &size);
    return size;
}

static size_t pml_ucx_generic_datatype_pack(void *state, size_t offset,
                                            void *dest, size_t max_length)
{
    mca_pml_ucx_convertor_t *convertor = state;
    uint32_t iov_count;
    struct iovec iov;
    size_t length;

    iov_count    = 1;
    iov.iov_base = dest;
    iov.iov_len  = max_length;

    opal_convertor_set_position(&convertor->opal_conv, &offset);
    length = max_length;
    opal_convertor_pack(&convertor->opal_conv, &iov, &iov_count, &length);
    return length;
}

static ucs_status_t pml_ucx_generic_datatype_unpack(void *state, size_t offset,
                                                    const void *src, size_t length)
{
    mca_pml_ucx_convertor_t *convertor = state;

    uint32_t iov_count;
    struct iovec iov;
    opal_convertor_t conv;

    iov_count    = 1;
    iov.iov_base = (void*)src;
    iov.iov_len  = length;

    /* in case if unordered message arrived - create separate convertor to
     * unpack data. */
    if (offset != convertor->offset) {
        OBJ_CONSTRUCT(&conv, opal_convertor_t);
        opal_convertor_copy_and_prepare_for_recv(ompi_proc_local_proc->super.proc_convertor,
                                                 &convertor->datatype->super,
                                                 convertor->opal_conv.count,
                                                 convertor->opal_conv.pBaseBuf, 0,
                                                 &conv);
        opal_convertor_set_position(&conv, &offset);
        opal_convertor_unpack(&conv, &iov, &iov_count, &length);
        opal_convertor_cleanup(&conv);
        OBJ_DESTRUCT(&conv);
        /* permanently switch to un-ordered mode */
        convertor->offset = 0;
    } else {
        opal_convertor_unpack(&convertor->opal_conv, &iov, &iov_count, &length);
        convertor->offset += length;
    }
    return UCS_OK;
}

static void pml_ucx_generic_datatype_finish(void *state)
{
    mca_pml_ucx_convertor_t *convertor = state;

    opal_convertor_cleanup(&convertor->opal_conv);
    OMPI_DATATYPE_RELEASE(convertor->datatype);
    PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.convs, &convertor->super);
}

static ucp_generic_dt_ops_t pml_ucx_generic_datatype_ops = {
    .start_pack   = pml_ucx_generic_datatype_start_pack,
    .start_unpack = pml_ucx_generic_datatype_start_unpack,
    .packed_size  = pml_ucx_generic_datatype_packed_size,
    .pack         = pml_ucx_generic_datatype_pack,
    .unpack       = pml_ucx_generic_datatype_unpack,
    .finish       = pml_ucx_generic_datatype_finish
};

int mca_pml_ucx_datatype_attr_del_fn(ompi_datatype_t* datatype, int keyval,
                                     void *attr_val, void *extra)
{
    ucp_datatype_t ucp_datatype = (ucp_datatype_t)attr_val;

#ifdef HAVE_UCP_REQUEST_PARAM_T
    free((void*)datatype->pml_data);
#else
    PML_UCX_ASSERT((uint64_t)ucp_datatype == datatype->pml_data);
#endif
    ucp_dt_destroy(ucp_datatype);
    datatype->pml_data = PML_UCX_DATATYPE_INVALID;
    return OMPI_SUCCESS;
}

__opal_attribute_always_inline__
static inline int mca_pml_ucx_datatype_is_contig(ompi_datatype_t *datatype)
{
    ptrdiff_t lb;

    ompi_datatype_type_lb(datatype, &lb);

    return (datatype->super.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS) &&
           (datatype->super.flags & OPAL_DATATYPE_FLAG_NO_GAPS) &&
           (lb == 0);
}

#ifdef HAVE_UCP_REQUEST_PARAM_T
__opal_attribute_always_inline__ static inline
pml_ucx_datatype_t *mca_pml_ucx_init_nbx_datatype(ompi_datatype_t *datatype,
                                                  ucp_datatype_t ucp_datatype,
                                                  size_t size)
{
    pml_ucx_datatype_t *pml_datatype;
    int is_contig_pow2;

    pml_datatype = malloc(sizeof(*pml_datatype));
    if (pml_datatype == NULL) {
        PML_UCX_ERROR("Failed to allocate datatype structure");
        ompi_mpi_abort(&ompi_mpi_comm_world.comm, 1);
    }

    pml_datatype->datatype                    = ucp_datatype;
    pml_datatype->op_param.send.op_attr_mask  = UCP_OP_ATTR_FIELD_CALLBACK;
    pml_datatype->op_param.send.cb.send       = mca_pml_ucx_send_nbx_completion;
    pml_datatype->op_param.bsend.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    pml_datatype->op_param.bsend.cb.send      = mca_pml_ucx_bsend_nbx_completion;
    pml_datatype->op_param.recv.op_attr_mask  = UCP_OP_ATTR_FIELD_CALLBACK |
                                                UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    pml_datatype->op_param.recv.cb.recv       = mca_pml_ucx_recv_nbx_completion;

    is_contig_pow2 = mca_pml_ucx_datatype_is_contig(datatype) &&
                     (size && !(size & (size - 1))); /* is_pow2(size) */
    if (is_contig_pow2) {
        pml_datatype->size_shift = (int)(log(size) / log(2.0)); /* log2(size) */
    } else {
        pml_datatype->size_shift = 0;
        PML_UCX_DATATYPE_SET_VALUE(pml_datatype, op_attr_mask |= UCP_OP_ATTR_FIELD_DATATYPE);
        PML_UCX_DATATYPE_SET_VALUE(pml_datatype, datatype = ucp_datatype);
    }

    return pml_datatype;
}
#endif

ucp_datatype_t mca_pml_ucx_init_datatype(ompi_datatype_t *datatype)
{
    size_t size = 0; /* init to suppress compiler warning */
    ucp_datatype_t ucp_datatype;
    ucs_status_t status;
    int ret;

    if (mca_pml_ucx_datatype_is_contig(datatype)) {
        ompi_datatype_type_size(datatype, &size);
        PML_UCX_ASSERT(size > 0);
        ucp_datatype = ucp_dt_make_contig(size);
        goto out;
    }

    status = ucp_dt_create_generic(&pml_ucx_generic_datatype_ops,
                                   datatype, &ucp_datatype);
    if (status != UCS_OK) {
        PML_UCX_ERROR("Failed to create UCX datatype for %s", datatype->name);
        ompi_mpi_abort(&ompi_mpi_comm_world.comm, 1);
    }

    /* Add custom attribute, to clean up UCX resources when OMPI datatype is
     * released.
     */
    if (ompi_datatype_is_predefined(datatype)) {
        PML_UCX_ASSERT(datatype->id < OMPI_DATATYPE_MAX_PREDEFINED);
        ompi_pml_ucx.predefined_types[datatype->id] = ucp_datatype;
    } else {
        ret = ompi_attr_set_c(TYPE_ATTR, datatype, &datatype->d_keyhash,
                              ompi_pml_ucx.datatype_attr_keyval,
                              (void*)ucp_datatype, false);
        if (ret != OMPI_SUCCESS) {
            PML_UCX_ERROR("Failed to add UCX datatype attribute for %s: %d",
                          datatype->name, ret);
            ompi_mpi_abort(&ompi_mpi_comm_world.comm, 1);
        }
    }
out:
    PML_UCX_VERBOSE(7, "created generic UCX datatype 0x%"PRIx64, ucp_datatype)

#ifdef HAVE_UCP_REQUEST_PARAM_T
    UCS_STATIC_ASSERT(sizeof(datatype->pml_data) >= sizeof(pml_ucx_datatype_t*));
    datatype->pml_data = (uint64_t)mca_pml_ucx_init_nbx_datatype(datatype,
                                                                 ucp_datatype,
                                                                 size);
#else
    datatype->pml_data = ucp_datatype;
#endif

    return ucp_datatype;
}

static void mca_pml_ucx_convertor_construct(mca_pml_ucx_convertor_t *convertor)
{
    OBJ_CONSTRUCT(&convertor->opal_conv, opal_convertor_t);
}

static void mca_pml_ucx_convertor_destruct(mca_pml_ucx_convertor_t *convertor)
{
    OBJ_DESTRUCT(&convertor->opal_conv);
}

OBJ_CLASS_INSTANCE(mca_pml_ucx_convertor_t,
                   opal_free_list_item_t,
                   mca_pml_ucx_convertor_construct,
                   mca_pml_ucx_convertor_destruct);
