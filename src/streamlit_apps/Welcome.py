import streamlit as st
import sys

sys.path.append("src")


def main() -> None:
    '''
    Multipage app
    Pages are in /pages folder
    '''
    st.set_page_config('Laptime Simulator')
    st.markdown(
        '''
            **Laptime Simulator\n
            by: Jan-Jaap van de Velde

            https://github.com/Jan-Jaap/laptime-sim
        '''
    )


if __name__ == "__main__":
    main()
