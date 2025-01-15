import sys
import streamlit as st

sys.path.append("src")


def main() -> None:
    """
    Multipage app
    Pages are in /pages folder
    """
    st.set_page_config("Laptime Simulator")
    st.markdown(
        """
            **Laptime Simulator
            \n
            by: Jan-Jaap van de Velde

            https://github.com/Jan-Jaap/laptime-sim

            I'm no math wizard.
            Optimization of laptimes is done by bruteforce trying different lines and using the fastest line found.
            By trying thousands of different options, we can come close (enough) to the optimum raceline.
            Some would call this AI...\n 
            Please use the menu options in the sidebar\n
            If the results viewer doesn't work, maybe create some results by running optimizations first :-)

        """
    )


if __name__ == "__main__":
    main()
